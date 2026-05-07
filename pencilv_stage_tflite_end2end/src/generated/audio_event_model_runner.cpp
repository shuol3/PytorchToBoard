// 本文件由部署脚本自动生成，负责在板端复现音频前处理并调用 TFLM 推理。
#include "generated/audio_event_model_runner.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>

#include <arm_math.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "generated/audio_event_model_config.h"
#include "generated/audio_event_model_data.h"

namespace audio_event_model_runner {
namespace {

// 这里集中定义生成 runner 的前端尺寸、量化辅助常量和固定缓冲区。
constexpr int kWindowSampleCount =
    (audio_event_model_config::kSampleRateHz *
     audio_event_model_config::kCaptureWindowMs) / 1000;
constexpr int kFftLength = audio_event_model_config::kFftLength;
constexpr int kWindowLenSamples =
    (audio_event_model_config::kSampleRateHz *
     audio_event_model_config::kFrameLengthMs) / 1000;
constexpr int kAnalysisFrameSamples = kFftLength;
constexpr int kFrameStrideSamples =
    (audio_event_model_config::kSampleRateHz *
     audio_event_model_config::kFrameStrideMs) / 1000;
constexpr int kFftBinCount = (kFftLength / 2) + 1;
constexpr int kBaseMelBinCount = audio_event_model_config::kFrontendMelBinCount;
constexpr int kBaseFeatureFrameCount = audio_event_model_config::kFrontendFeatureFrameCount;
constexpr int kBaseFeatureElementCount = audio_event_model_config::kFrontendFeatureElementCount;
constexpr int kWindowPadLeft =
    (kAnalysisFrameSamples > kWindowLenSamples)
        ? ((kAnalysisFrameSamples - kWindowLenSamples) / 2)
        : 0;
constexpr int kFrameFftCopySamples =
    (kWindowLenSamples < kAnalysisFrameSamples) ? kWindowLenSamples : kAnalysisFrameSamples;
constexpr int kMelFilterCoefficientCount = 613;
constexpr size_t kTensorArenaSize = 122880;
constexpr float kPcmScale = 1.0f / 32768.0f;
constexpr float kNaturalLogToDbScale = 4.342944819032518f;
constexpr float kDefaultDecisionThreshold = 0.5f;
constexpr int kMaxProfiledOps = 32;
constexpr uint32_t kInvokeProfileReportAfterInvocations = 5;

enum class InputLayout {
    kFlat,
    kMatrix,
    kNhwc,
    kNchw,
    kUnsupported,
};

struct ModelInputSpec {
    InputLayout layout = InputLayout::kUnsupported;
    int rows = 0;
    int cols = 0;
    int channels = 0;
    int element_count = 0;
};

// 杩欎釜缁撴瀯鐢ㄤ簬璁板綍鍗曟 Invoke 鍐呴儴姣忎釜绠楀瓙鐨勫懆鏈熸暟锛屾柟渚胯瘖鏂ā鍨嬫湰浣撶摱棰堛€?
struct ProfiledInvokeEvent {
    const char *tag = "";
    uint32_t start_cycles = 0;
    uint32_t elapsed_cycles = 0;
};

// 浣跨敤 Zephyr 楂樼簿搴︾‖浠跺懆鏈熻鏁板櫒瀹炵幇 TFLM profiler 鎺ュ彛锛屽彧瀵筁nvoke 鍐呴儴绠楀瓙鎵撶偣銆?
class InvokeProfiler final : public tflite::MicroProfilerInterface {
 public:
    void Reset()
    {
        event_count_ = 0;
        overflowed_ = false;
    }

    uint32_t BeginEvent(const char *tag) override
    {
        if (event_count_ >= kMaxProfiledOps) {
            overflowed_ = true;
            return static_cast<uint32_t>(kMaxProfiledOps);
        }

        events_[event_count_].tag = tag;
        events_[event_count_].start_cycles = k_cycle_get_32();
        events_[event_count_].elapsed_cycles = 0;
        return static_cast<uint32_t>(event_count_++);
    }

    void EndEvent(uint32_t event_handle) override
    {
        if (event_handle >= static_cast<uint32_t>(event_count_)) {
            return;
        }
        events_[event_handle].elapsed_cycles =
            k_cycle_get_32() - events_[event_handle].start_cycles;
    }

    int event_count() const
    {
        return event_count_;
    }

    bool overflowed() const
    {
        return overflowed_;
    }

    const ProfiledInvokeEvent &event(int index) const
    {
        return events_[index];
    }

 private:
    ProfiledInvokeEvent events_[kMaxProfiledOps] = {};
    int event_count_ = 0;
    bool overflowed_ = false;
};

// 这些常量由 PC 侧离线生成，保证板端与导出参数严格一致。
alignas(16) static const float g_window[kWindowLenSamples] = {
    0.0f, 6.168375693960115e-05f, 0.00024671980645507574f, 0.0005550624919123948f, 0.0009866358013823628f, 0.0015413331566378474f, 0.002219017595052719f, 0.003019522177055478f,
    0.003942649345844984f, 0.0049881711602211f, 0.006155829876661301f, 0.007445336785167456f, 0.008856374770402908f, 0.010388595052063465f, 0.012041619047522545f, 0.01381503976881504f,
    0.01570841856300831f, 0.01772129163146019f, 0.01985315792262554f, 0.02210349217057228f, 0.024471741169691086f, 0.026957320049405098f, 0.02955961599946022f, 0.03227798640727997f,
    0.03511175885796547f, 0.03806023299694061f, 0.0411226861178875f, 0.044298361986875534f, 0.04758647456765175f, 0.050986211746931076f, 0.05449673905968666f, 0.05811718478798866f,
    0.06184665858745575f, 0.0656842440366745f, 0.06962898373603821f, 0.07367991656064987f, 0.0778360366821289f, 0.08209631592035294f, 0.08645971119403839f, 0.09092514216899872f,
    0.09549150615930557f, 0.10015767067670822f, 0.10492249578237534f, 0.109784796833992f, 0.11474338173866272f, 0.11979701370000839f, 0.12494446337223053f, 0.13018445670604706f,
    0.13551568984985352f, 0.14093685150146484f, 0.1464466154575348f, 0.15204359591007233f, 0.15772645175457f, 0.16349373757839203f, 0.16934406757354736f, 0.175275981426239f,
    0.1812880039215088f, 0.1873786747455597f, 0.19354647397994995f, 0.1997898817062378f, 0.20610737800598145f, 0.21249736845493317f, 0.21895831823349f, 0.2254885882139206f,
    0.23208659887313843f, 0.23875071108341217f, 0.2454792857170105f, 0.2522706687450409f, 0.25912317633628845f, 0.2660350799560547f, 0.2730047404766083f, 0.28003042936325073f,
    0.2871103584766388f, 0.29424282908439636f, 0.3014260530471802f, 0.30865827202796936f, 0.31593772768974304f, 0.3232625722885132f, 0.3306310474872589f, 0.3380413055419922f,
    0.345491498708725f, 0.352979838848114f, 0.3605044484138489f, 0.36806347966194153f, 0.37565505504608154f, 0.3832773268222809f, 0.39092838764190674f, 0.3986063599586487f,
    0.4063093364238739f, 0.414035439491272f, 0.4217827618122101f, 0.4295493960380554f, 0.4373333752155304f, 0.445132851600647f, 0.45294585824012756f, 0.46077045798301697f,
    0.4686047434806824f, 0.4764467775821686f, 0.4842946231365204f, 0.4921463429927826f, 0.5f, 0.5078536868095398f, 0.515705406665802f, 0.5235532522201538f,
    0.5313952565193176f, 0.5392295718193054f, 0.5470541715621948f, 0.554867148399353f, 0.5626665949821472f, 0.5704506039619446f, 0.5782172083854675f, 0.585964560508728f,
    0.5936906337738037f, 0.6013936400413513f, 0.6090716123580933f, 0.6167227029800415f, 0.6243449449539185f, 0.6319365501403809f, 0.6394955515861511f, 0.647020161151886f,
    0.6545084714889526f, 0.6619586944580078f, 0.6693689823150635f, 0.6767374277114868f, 0.6840623021125793f, 0.6913416981697083f, 0.6985739469528198f, 0.705757200717926f,
    0.7128896713256836f, 0.7199695706367493f, 0.7269952297210693f, 0.7339649200439453f, 0.7408768534660339f, 0.7477293610572815f, 0.7545207142829895f, 0.761249303817749f,
    0.7679134011268616f, 0.7745113968849182f, 0.78104168176651f, 0.787502646446228f, 0.7938926219940186f, 0.8002101182937622f, 0.80645352602005f, 0.8126213550567627f,
    0.8187119960784912f, 0.824724018573761f, 0.8306559324264526f, 0.8365062475204468f, 0.8422735333442688f, 0.8479564189910889f, 0.8535534143447876f, 0.8590631484985352f,
    0.8644843101501465f, 0.8698155283927917f, 0.8750555515289307f, 0.8802030086517334f, 0.8852566480636597f, 0.8902152180671692f, 0.8950775265693665f, 0.8998423218727112f,
    0.9045084714889526f, 0.9090748429298401f, 0.9135403037071228f, 0.9179036617279053f, 0.9221639633178711f, 0.9263200759887695f, 0.9303709864616394f, 0.9343157410621643f,
    0.9381533265113831f, 0.9418827891349792f, 0.9455032348632812f, 0.9490137696266174f, 0.9524134993553162f, 0.9557016491889954f, 0.9588773250579834f, 0.9619397521018982f,
    0.9648882150650024f, 0.9677219986915588f, 0.9704403877258301f, 0.9730426669120789f, 0.9755282402038574f, 0.977896511554718f, 0.9801468253135681f, 0.9822787046432495f,
    0.9842915534973145f, 0.9861849546432495f, 0.9879583716392517f, 0.9896113872528076f, 0.9911436438560486f, 0.9925546646118164f, 0.9938441514968872f, 0.9950118064880371f,
    0.9960573315620422f, 0.9969804883003235f, 0.997780978679657f, 0.9984586834907532f, 0.999013364315033f, 0.9994449615478516f, 0.9997532963752747f, 0.9999383091926575f,
    1.0f, 0.9999383091926575f, 0.9997532963752747f, 0.9994449615478516f, 0.999013364315033f, 0.9984586834907532f, 0.997780978679657f, 0.9969804883003235f,
    0.9960573315620422f, 0.9950118064880371f, 0.9938441514968872f, 0.9925546646118164f, 0.9911436438560486f, 0.9896113872528076f, 0.9879583716392517f, 0.9861849546432495f,
    0.9842915534973145f, 0.9822787046432495f, 0.9801468253135681f, 0.977896511554718f, 0.9755282402038574f, 0.9730426669120789f, 0.9704403877258301f, 0.9677219986915588f,
    0.9648882150650024f, 0.9619397521018982f, 0.9588773250579834f, 0.9557016491889954f, 0.9524134993553162f, 0.9490137696266174f, 0.9455032348632812f, 0.9418827891349792f,
    0.9381533265113831f, 0.9343157410621643f, 0.9303709864616394f, 0.9263200759887695f, 0.9221639633178711f, 0.9179036617279053f, 0.9135403037071228f, 0.9090748429298401f,
    0.9045084714889526f, 0.8998423218727112f, 0.8950775265693665f, 0.8902152180671692f, 0.8852566480636597f, 0.8802030086517334f, 0.8750555515289307f, 0.8698155283927917f,
    0.8644843101501465f, 0.8590631484985352f, 0.8535534143447876f, 0.8479564189910889f, 0.8422735333442688f, 0.8365062475204468f, 0.8306559324264526f, 0.824724018573761f,
    0.8187119960784912f, 0.8126213550567627f, 0.80645352602005f, 0.8002101182937622f, 0.7938926219940186f, 0.787502646446228f, 0.78104168176651f, 0.7745113968849182f,
    0.7679134011268616f, 0.761249303817749f, 0.7545207142829895f, 0.7477293610572815f, 0.7408768534660339f, 0.7339649200439453f, 0.7269952297210693f, 0.7199695706367493f,
    0.7128896713256836f, 0.705757200717926f, 0.6985739469528198f, 0.6913416981697083f, 0.6840623021125793f, 0.6767374277114868f, 0.6693689823150635f, 0.6619586944580078f,
    0.6545084714889526f, 0.647020161151886f, 0.6394955515861511f, 0.6319365501403809f, 0.6243449449539185f, 0.6167227029800415f, 0.6090716123580933f, 0.6013936400413513f,
    0.5936906337738037f, 0.585964560508728f, 0.5782172083854675f, 0.5704506039619446f, 0.5626665949821472f, 0.554867148399353f, 0.5470541715621948f, 0.5392295718193054f,
    0.5313952565193176f, 0.5235532522201538f, 0.515705406665802f, 0.5078536868095398f, 0.5f, 0.4921463429927826f, 0.4842946231365204f, 0.4764467775821686f,
    0.4686047434806824f, 0.46077045798301697f, 0.45294585824012756f, 0.445132851600647f, 0.4373333752155304f, 0.4295493960380554f, 0.4217827618122101f, 0.414035439491272f,
    0.4063093364238739f, 0.3986063599586487f, 0.39092838764190674f, 0.3832773268222809f, 0.37565505504608154f, 0.36806347966194153f, 0.3605044484138489f, 0.352979838848114f,
    0.345491498708725f, 0.3380413055419922f, 0.3306310474872589f, 0.3232625722885132f, 0.31593772768974304f, 0.30865827202796936f, 0.3014260530471802f, 0.29424282908439636f,
    0.2871103584766388f, 0.28003042936325073f, 0.2730047404766083f, 0.2660350799560547f, 0.25912317633628845f, 0.2522706687450409f, 0.2454792857170105f, 0.23875071108341217f,
    0.23208659887313843f, 0.2254885882139206f, 0.21895831823349f, 0.21249736845493317f, 0.20610737800598145f, 0.1997898817062378f, 0.19354647397994995f, 0.1873786747455597f,
    0.1812880039215088f, 0.175275981426239f, 0.16934406757354736f, 0.16349373757839203f, 0.15772645175457f, 0.15204359591007233f, 0.1464466154575348f, 0.14093685150146484f,
    0.13551568984985352f, 0.13018445670604706f, 0.12494446337223053f, 0.11979701370000839f, 0.11474338173866272f, 0.109784796833992f, 0.10492249578237534f, 0.10015767067670822f,
    0.09549150615930557f, 0.09092514216899872f, 0.08645971119403839f, 0.08209631592035294f, 0.0778360366821289f, 0.07367991656064987f, 0.06962898373603821f, 0.0656842440366745f,
    0.06184665858745575f, 0.05811718478798866f, 0.05449673905968666f, 0.050986211746931076f, 0.04758647456765175f, 0.044298361986875534f, 0.0411226861178875f, 0.03806023299694061f,
    0.03511175885796547f, 0.03227798640727997f, 0.02955961599946022f, 0.026957320049405098f, 0.024471741169691086f, 0.02210349217057228f, 0.01985315792262554f, 0.01772129163146019f,
    0.01570841856300831f, 0.01381503976881504f, 0.012041619047522545f, 0.010388595052063465f, 0.008856374770402908f, 0.007445336785167456f, 0.006155829876661301f, 0.0049881711602211f,
    0.003942649345844984f, 0.003019522177055478f, 0.002219017595052719f, 0.0015413331566378474f, 0.0009866358013823628f, 0.0005550624919123948f, 0.00024671980645507574f, 6.168375693960115e-05f
};
alignas(16) static const uint16_t g_mel_filter_pos[kBaseMelBinCount] = {
    193, 197, 201, 205, 209, 214, 218, 223, 227, 232, 237, 242,
    247, 252, 257, 262, 268, 273, 279, 284, 290, 296, 302, 308,
    314, 320, 326, 333, 339, 346, 353, 359, 366, 374, 381, 388,
    396, 403, 411, 419, 427, 435, 444, 452, 461, 469, 478, 487
};
alignas(16) static const uint16_t g_mel_filter_len[kBaseMelBinCount] = {
    8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10,
    10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12,
    12, 13, 13, 13, 14, 13, 13, 15, 15, 14, 15, 15,
    15, 16, 16, 16, 17, 17, 17, 17, 17, 18, 19, 19
};
alignas(16) static const float g_mel_filter_coefs[kMelFilterCoefficientCount] = {
    0.24323303997516632f, 0.48646607995033264f, 0.7296991348266602f, 0.9729321599006653f, 0.7875237464904785f, 0.5484416484832764f, 0.30935949087142944f, 0.07027735561132431f,
    0.2124762386083603f, 0.451558381319046f, 0.6906405091285706f, 0.9297226667404175f, 0.8340759873390198f, 0.5990738868713379f, 0.3640718162059784f, 0.1290697604417801f,
    0.16592402756214142f, 0.4009261131286621f, 0.6359281539916992f, 0.8709302544593811f, 1.105932354927063f, 0.6648838520050049f, 0.43389222025871277f, 0.20290060341358185f,
    0.104124516248703f, 0.3351161479949951f, 0.5661077499389648f, 0.7970994114875793f, 1.0280910730361938f, 0.7453387379646301f, 0.5182890892028809f, 0.291239470243454f,
    0.0641898363828659f, 0.027611641213297844f, 0.2546612620353699f, 0.48171091079711914f, 0.7087605595588684f, 0.9358101487159729f, 0.8399195075035095f, 0.6167445778846741f,
    0.3935697078704834f, 0.17039479315280914f, 0.16008049249649048f, 0.38325539231300354f, 0.6064302921295166f, 0.829605221748352f, 1.0527801513671875f, 0.7287542819976807f,
    0.5093880295753479f, 0.29002171754837036f, 0.07065540552139282f, 0.0518793910741806f, 0.27124568819999695f, 0.4906120002269745f, 0.7099782824516296f, 0.9293445944786072f,
    0.8538269400596619f, 0.6382042169570923f, 0.4225815534591675f, 0.2069588601589203f, 0.14617305994033813f, 0.3617957532405853f, 0.5774184465408325f, 0.7930411696434021f,
    1.0086638927459717f, 0.779541015625f, 0.5675980448722839f, 0.35565510392189026f, 0.143712118268013f, 0.008515982888638973f, 0.2204589545726776f, 0.4324019253253937f,
    0.6443449258804321f, 0.8562878966331482f, 1.0682308673858643f, 0.7246074676513672f, 0.5162814259529114f, 0.30795538425445557f, 0.09962934255599976f, 0.06706646084785461f,
    0.2753925025463104f, 0.48371854424476624f, 0.6920446157455444f, 0.9003706574440002f, 0.8931582570075989f, 0.688387393951416f, 0.48361656069755554f, 0.2788456976413727f,
    0.07407484948635101f, 0.10684174299240112f, 0.311612606048584f, 0.5163834691047668f, 0.7211542725563049f, 0.9259251356124878f, 0.8715344071388245f, 0.6702580451965332f,
    0.4689817428588867f, 0.26770541071891785f, 0.06642909348011017f, 0.12846560776233673f, 0.3297419250011444f, 0.5310182571411133f, 0.7322945594787598f, 0.933570921421051f,
    0.8674539923667908f, 0.6696125864982605f, 0.47177115082740784f, 0.2739297151565552f, 0.07608827948570251f, 0.13254599273204803f, 0.3303874135017395f, 0.5282288789749146f,
    0.7260702848434448f, 0.9239117503166199f, 0.8803246021270752f, 0.6858594417572021f, 0.4913943111896515f, 0.29692915081977844f, 0.1024639829993248f, 0.11967537552118301f,
    0.31414052844047546f, 0.5086057186126709f, 0.703070878982544f, 0.897536039352417f, 1.09200119972229f, 0.7184223532676697f, 0.5272758603096008f, 0.336129367351532f,
    0.14498284459114075f, 0.09043112397193909f, 0.2815776467323303f, 0.47272413969039917f, 0.663870632648468f, 0.8550171256065369f, 1.0461636781692505f, 0.7667396664619446f,
    0.5788551568984985f, 0.39097070693969727f, 0.20308621227741241f, 0.015201722271740437f, 0.045375850051641464f, 0.23326033353805542f, 0.4211448132991791f, 0.6090292930603027f,
    0.7969138026237488f, 0.98479825258255f, 0.830264151096344f, 0.6455860137939453f, 0.460907906293869f, 0.27622976899147034f, 0.09155162423849106f, 0.1697358340024948f,
    0.3544139862060547f, 0.5390921235084534f, 0.723770260810852f, 0.9084483981132507f, 0.9084627628326416f, 0.7269362211227417f, 0.5454097390174866f, 0.36388325691223145f,
    0.18235674500465393f, 0.0008302489877678454f, 0.09153725206851959f, 0.2730637490749359f, 0.4545902609825134f, 0.6361167430877686f, 0.8176432251930237f, 0.9991697669029236f,
    0.8223874568939209f, 0.6439588069915771f, 0.465530127286911f, 0.28710147738456726f, 0.10867282748222351f, 0.1776125729084015f, 0.35604122281074524f, 0.5344698429107666f,
    0.7128984928131104f, 0.8913271427154541f, 1.0697557926177979f, 0.7560509443283081f, 0.58066725730896f, 0.4052836000919342f, 0.22989992797374725f, 0.0545162633061409f,
    0.06856539845466614f, 0.2439490705728531f, 0.41933274269104004f, 0.5947164297103882f, 0.7701000571250916f, 0.9454837441444397f, 0.8811952471733093f, 0.7088046073913574f,
    0.5364139676094055f, 0.3640233278274536f, 0.19163267314434052f, 0.019242022186517715f, 0.11880473792552948f, 0.2911953926086426f, 0.4635860323905945f, 0.6359766721725464f,
    0.8083673119544983f, 0.9807579517364502f, 0.8494649529457092f, 0.6800162196159363f, 0.5105675458908081f, 0.34111881256103516f, 0.1716701090335846f, 0.002221401548013091f,
    0.15053506195545197f, 0.3199837803840637f, 0.4894324839115143f, 0.6588811874389648f, 0.8283299207687378f, 0.997778594493866f, 0.8356265425682068f, 0.6690695285797119f,
    0.5025125741958618f, 0.33595559000968933f, 0.16939863562583923f, 0.0028416600544005632f, 0.1643734872341156f, 0.3309304416179657f, 0.4974874258041382f, 0.6640443801879883f,
    0.8306013941764832f, 0.9971583485603333f, 0.8390786051750183f, 0.6753640174865723f, 0.5116494297981262f, 0.3479348123073578f, 0.18422023952007294f, 0.020505651831626892f,
    0.16092142462730408f, 0.3246360123157501f, 0.48835060000419617f, 0.6520651578903198f, 0.8157797455787659f, 0.9794943332672119f, 0.8592349886894226f, 0.6983143091201782f,
    0.5373935699462891f, 0.3764728903770447f, 0.2155521810054779f, 0.05463147163391113f, 0.1407649964094162f, 0.3016856908798218f, 0.46260640025138855f, 0.6235271096229553f,
    0.7844478487968445f, 0.9453685283660889f, 0.8955246806144714f, 0.7373501658439636f, 0.5791756510734558f, 0.421001136302948f, 0.2628266215324402f, 0.10465212166309357f,
    0.10447534918785095f, 0.26264986395835876f, 0.4208243489265442f, 0.578998863697052f, 0.7371733784675598f, 0.8953478932380676f, 1.0535223484039307f, 0.7919158339500427f,
    0.636440634727478f, 0.4809654951095581f, 0.3254903256893158f, 0.1700151562690735f, 0.014539978466928005f, 0.052608996629714966f, 0.20808416604995728f, 0.3635593354701996f,
    0.5190345048904419f, 0.6745097041130066f, 0.8299848437309265f, 0.9854600429534912f, 0.8614699244499207f, 0.7086480259895325f, 0.5558261275291443f, 0.4030042290687561f,
    0.2501823306083679f, 0.09736043214797974f, 0.13853006064891815f, 0.29135194420814514f, 0.4441738724708557f, 0.5969957709312439f, 0.7498176693916321f, 0.9026395678520203f,
    1.0554614067077637f, 0.795271098613739f, 0.645057201385498f, 0.4948432743549347f, 0.3446293771266937f, 0.19441545009613037f, 0.04420154169201851f, 0.05451498553156853f,
    0.204728901386261f, 0.35494279861450195f, 0.5051566958427429f, 0.6553706526756287f, 0.8055845499038696f, 0.9557984471321106f, 0.8957967758178711f, 0.7481463551521301f,
    0.6004959344863892f, 0.4528455138206482f, 0.30519506335258484f, 0.15754464268684387f, 0.009894216433167458f, 0.10420320928096771f, 0.2518536448478699f, 0.39950406551361084f,
    0.5471544861793518f, 0.6948049068450928f, 0.8424553275108337f, 0.9901058077812195f, 0.8645946979522705f, 0.7194640040397644f, 0.5743333101272583f, 0.4292025864124298f,
    0.2840718924999237f, 0.1389412134885788f, 0.13540533185005188f, 0.280536025762558f, 0.4256667196750641f, 0.5707973837852478f, 0.7159280776977539f, 0.86105877161026f,
    1.0061894655227661f, 0.8512622117996216f, 0.7086082100868225f, 0.5659542679786682f, 0.42330029606819153f, 0.28064635396003723f, 0.13799239695072174f, 0.006083858665078878f,
    0.1487378180027008f, 0.2913917899131775f, 0.4340457320213318f, 0.5766996741294861f, 0.7193536758422852f, 0.8620076179504395f, 1.0046615600585938f, 0.8551985025405884f,
    0.7149789929389954f, 0.5747595429420471f, 0.4345400333404541f, 0.2943205237388611f, 0.15410104393959045f, 0.013881557621061802f, 0.0045820134691894054f, 0.14480149745941162f,
    0.28502100706100464f, 0.42524048686027527f, 0.5654599666595459f, 0.7056794762611389f, 0.8458989262580872f, 0.9861184358596802f, 0.8758180737495422f, 0.7379915118217468f,
    0.6001649498939514f, 0.462338387966156f, 0.3245118260383606f, 0.18668526411056519f, 0.04885869100689888f, 0.12418190389871597f, 0.2620084583759308f, 0.3998350501060486f,
    0.537661612033844f, 0.6754881739616394f, 0.8133147358894348f, 0.9511412978172302f, 0.9125503897666931f, 0.7770759463310242f, 0.6416014432907104f, 0.5061269402503967f,
    0.3706524968147278f, 0.23517800867557526f, 0.09970352053642273f, 0.0874495878815651f, 0.22292406857013702f, 0.35839855670928955f, 0.4938730299472809f, 0.6293475031852722f,
    0.7648220062255859f, 0.9002964496612549f, 1.0357710123062134f, 0.8316769599914551f, 0.6985144019126892f, 0.5653519034385681f, 0.43218934535980225f, 0.29902681708335876f,
    0.16586428880691528f, 0.032701753079891205f, 0.035160504281520844f, 0.16832304000854492f, 0.3014855682849884f, 0.4346480965614319f, 0.5678106546401978f, 0.7009731531143188f,
    0.8341357111930847f, 0.9672982692718506f, 0.9012536406517029f, 0.7703635692596436f, 0.639473557472229f, 0.5085834860801697f, 0.3776934742927551f, 0.246803417801857f,
    0.11591338366270065f, 0.09874636679887772f, 0.22963640093803406f, 0.360526442527771f, 0.49141648411750793f, 0.6223065257072449f, 0.7531965970993042f, 0.8840866088867188f,
    1.0149766206741333f, 0.856622576713562f, 0.7279662489891052f, 0.5993099212646484f, 0.47065359354019165f, 0.34199726581573486f, 0.21334093809127808f, 0.0846845954656601f,
    0.014721076935529709f, 0.1433774083852768f, 0.2720337510108948f, 0.40069007873535156f, 0.5293464064598083f, 0.6580027341842651f, 0.7866590619087219f, 0.9153153896331787f,
    1.0439717769622803f, 0.8303179144859314f, 0.7038571834564209f, 0.5773964524269104f, 0.4509356915950775f, 0.324474960565567f, 0.19801422953605652f, 0.07155348360538483f,
    0.043221332132816315f, 0.1696820706129074f, 0.2961428165435791f, 0.4226035475730896f, 0.5490642786026001f, 0.6755250096321106f, 0.8019858002662659f, 0.9284465312957764f,
    1.054907202720642f, 0.8217271566390991f, 0.6974245309829712f, 0.5731219053268433f, 0.44881927967071533f, 0.3245166838169098f, 0.20021404325962067f, 0.07591143250465393f,
    0.05397023633122444f, 0.17827285826206207f, 0.3025754690170288f, 0.42687809467315674f, 0.5511807203292847f, 0.6754833459854126f, 0.7997859716415405f, 0.9240885972976685f,
    1.0483912229537964f, 0.8302533030509949f, 0.7080720067024231f, 0.5858906507492065f, 0.4637093245983124f, 0.3415279984474182f, 0.21934667229652405f, 0.09716535359621048f,
    0.04756536707282066f, 0.16974669694900513f, 0.2919280230998993f, 0.41410934925079346f, 0.5362906455993652f, 0.6584720015525818f, 0.7806532979011536f, 0.9028346538543701f,
    1.025015950202942f, 0.8553146719932556f, 0.7352184653282166f, 0.6151222586631775f, 0.49502599239349365f, 0.3749297559261322f, 0.25483351945877075f, 0.1347372978925705f,
    0.014641055837273598f, 0.024589065462350845f, 0.144685298204422f, 0.26478153467178345f, 0.3848777711391449f, 0.5049740076065063f, 0.6250702142715454f, 0.7451664805412292f,
    0.8652626872062683f, 0.9853589534759521f, 0.8963444828987122f, 0.7782977223396301f, 0.6602510213851929f, 0.5422043204307556f, 0.4241575598716736f, 0.30611082911491394f,
    0.1880641132593155f, 0.07001738250255585f, 0.10365552455186844f, 0.22170224785804749f, 0.33974897861480713f, 0.4577957093715668f, 0.5758424401283264f, 0.6938891410827637f,
    0.8119359016418457f, 0.929982602596283f, 1.0480293035507202f, 0.8367581367492676f, 0.7207258939743042f, 0.6046937108039856f, 0.488661527633667f, 0.3726293444633484f,
    0.2565971314907074f, 0.1405649483203888f, 0.024532752111554146f, 0.04720969498157501f, 0.1632418930530548f, 0.2792740762233734f, 0.3953062891960144f, 0.511338472366333f,
    0.6273706555366516f, 0.7434028387069702f, 0.8594350814819336f, 0.9754672646522522f, 0.9100620746612549f, 0.7960100173950195f, 0.6819579601287842f, 0.5679059028625488f,
    0.45385387539863586f, 0.3398018479347229f, 0.22574980556964874f, 0.11169775575399399f, 0.0899379551410675f, 0.20398999750614166f, 0.3180420398712158f, 0.4320940673351288f,
    0.5461460947990417f, 0.6601981520652771f, 0.7742502093315125f, 0.8883022665977478f, 1.0023542642593384f, 0.8855802416801453f, 0.7734745144844055f, 0.6613688468933105f,
    0.5492631793022156f, 0.4371574819087982f, 0.32505181431770325f, 0.21294613182544708f, 0.10084044933319092f, 0.0023141049314290285f, 0.11441978812217712f, 0.2265254706144333f,
    0.33863115310668945f, 0.4507368206977844f, 0.5628424882888794f, 0.6749482154846191f, 0.7870538830757141f, 0.8991595506668091f, 1.0112652778625488f, 0.8787344694137573f,
    0.7685419321060181f, 0.6583493947982788f, 0.5481568574905396f, 0.4379643499851227f, 0.3277718126773834f, 0.21757927536964417f, 0.10738673806190491f, 0.011072980239987373f,
    0.12126551568508148f, 0.23145805299282074f, 0.3416505753993988f, 0.45184311270713806f, 0.5620356798171997f, 0.672228217124939f, 0.7824207544326782f, 0.8926132321357727f,
    1.0028058290481567f, 0.8889300227165222f, 0.7806180119514465f, 0.6723059415817261f, 0.5639939308166504f, 0.4556818902492523f, 0.34736984968185425f, 0.23905780911445618f,
    0.1307457685470581f, 0.02243373543024063f, 0.0027579141315072775f, 0.11106995493173599f, 0.21938198804855347f, 0.32769402861595154f, 0.4360060691833496f, 0.5443180799484253f,
    0.6526301503181458f, 0.7609421610832214f, 0.8692542314529419f, 0.9775662422180176f, 0.9155872464179993f, 0.8091236352920532f, 0.7026599645614624f, 0.5961963534355164f,
    0.4897327125072479f, 0.3832690715789795f, 0.27680546045303345f, 0.17034181952476501f, 0.06387817859649658f
};

alignas(16) static float g_fft_input[kFftLength];
alignas(16) static float g_fft_output[2 * kFftBinCount];
alignas(16) static float g_power_spectrum[kFftBinCount];
alignas(16) static float g_base_feature_buf[kBaseFeatureElementCount];
alignas(16) static uint8_t g_tensor_arena[kTensorArenaSize];

static arm_rfft_fast_instance_f32 g_fft_instance;
static const tflite::Model *g_model = nullptr;
static tflite::MicroInterpreter *g_interpreter = nullptr;
static TfLiteTensor *g_input = nullptr;
static TfLiteTensor *g_output = nullptr;
static ModelInputSpec g_input_spec;
static InvokeProfiler g_invoke_profiler;
static int g_output_count = 0;
static int g_model_operator_count = 0;
static bool g_frontend_initialized = false;
static bool g_frontend_init_failed = false;
static bool g_model_initialized = false;
static bool g_model_init_attempted = false;
static bool g_invoke_profile_reported = false;
static const char *g_last_error = "not initialized";
static uint32_t g_last_feature_ms = 0;
static uint32_t g_last_invoke_ms = 0;
static uint32_t g_invoke_profile_sample_count = 0;
static uint64_t g_invoke_profile_cycle_sums[kMaxProfiledOps] = {};

void set_last_error(const char *message)
{
    g_last_error = message;
}

uint32_t cycles_to_us(uint32_t cycles)
{
    const uint64_t cycle_hz = static_cast<uint64_t>(sys_clock_hw_cycles_per_sec());
    if (cycle_hz == 0U) {
        return 0U;
    }
    return static_cast<uint32_t>(
        ((static_cast<uint64_t>(cycles) * 1000000ULL) + (cycle_hz / 2ULL)) / cycle_hz);
}

const tflite::SubGraph *primary_subgraph()
{
    if (g_model == nullptr || g_model->subgraphs() == nullptr || g_model->subgraphs()->size() == 0) {
        return nullptr;
    }
    return g_model->subgraphs()->Get(0);
}

int flatbuffer_operator_count()
{
    const tflite::SubGraph *subgraph = primary_subgraph();
    if (subgraph == nullptr || subgraph->operators() == nullptr) {
        return 0;
    }
    return static_cast<int>(subgraph->operators()->size());
}

const tflite::Operator *flatbuffer_operator_at(int op_index)
{
    const tflite::SubGraph *subgraph = primary_subgraph();
    if (subgraph == nullptr || subgraph->operators() == nullptr) {
        return nullptr;
    }
    const int operator_count = static_cast<int>(subgraph->operators()->size());
    if (op_index < 0 || op_index >= operator_count) {
        return nullptr;
    }
    return subgraph->operators()->Get(op_index);
}

const tflite::OperatorCode *flatbuffer_operator_code(const tflite::Operator *op)
{
    if (g_model == nullptr || op == nullptr || g_model->operator_codes() == nullptr) {
        return nullptr;
    }
    const int opcode_index = op->opcode_index();
    const int opcode_count = static_cast<int>(g_model->operator_codes()->size());
    if (opcode_index < 0 || opcode_index >= opcode_count) {
        return nullptr;
    }
    return g_model->operator_codes()->Get(opcode_index);
}

const char *flatbuffer_operator_name(const tflite::Operator *op)
{
    const tflite::OperatorCode *opcode = flatbuffer_operator_code(op);
    if (opcode == nullptr) {
        return "UNKNOWN";
    }
    return tflite::EnumNameBuiltinOperator(opcode->builtin_code());
}

const tflite::Tensor *flatbuffer_tensor_at(int tensor_index)
{
    const tflite::SubGraph *subgraph = primary_subgraph();
    if (subgraph == nullptr || subgraph->tensors() == nullptr) {
        return nullptr;
    }
    const int tensor_count = static_cast<int>(subgraph->tensors()->size());
    if (tensor_index < 0 || tensor_index >= tensor_count) {
        return nullptr;
    }
    return subgraph->tensors()->Get(tensor_index);
}

void log_flatbuffer_tensor_shape(const tflite::Tensor *tensor)
{
    if (tensor == nullptr || tensor->shape() == nullptr) {
        printk("[]");
        return;
    }

    const int dim_count = static_cast<int>(tensor->shape()->size());
    printk("[");
    for (int dim_index = 0; dim_index < dim_count; ++dim_index) {
        if (dim_index > 0) {
            printk("x");
        }
        printk("%d", tensor->shape()->Get(dim_index));
    }
    printk("]");
}

void log_invoke_profile_summary(int event_count)
{
    if (g_invoke_profile_sample_count == 0) {
        return;
    }

    const int profiled_event_count = std::min(event_count, kMaxProfiledOps);
    const int mapped_event_count = std::min(profiled_event_count, g_model_operator_count);
    uint64_t total_avg_cycles = 0;
    for (int event_index = 0; event_index < profiled_event_count; ++event_index) {
        total_avg_cycles += g_invoke_profile_cycle_sums[event_index] / g_invoke_profile_sample_count;
    }

    printk("Invoke breakdown: avg over %u runs, cycle_hz=%u\n",
           g_invoke_profile_sample_count,
           static_cast<unsigned int>(sys_clock_hw_cycles_per_sec()));

    if (g_invoke_profiler.overflowed()) {
        printk("  profiler warning: event buffer overflowed, only first %d events kept\n",
               kMaxProfiledOps);
    }
    if (profiled_event_count != g_model_operator_count) {
        printk("  profiler note: event_count=%d, model_op_count=%d\n",
               profiled_event_count,
               g_model_operator_count);
    }

    for (int event_index = 0; event_index < profiled_event_count; ++event_index) {
        const uint32_t avg_cycles = static_cast<uint32_t>(
            g_invoke_profile_cycle_sums[event_index] / g_invoke_profile_sample_count);
        const uint32_t avg_us = cycles_to_us(avg_cycles);
        const uint32_t share_permille = (total_avg_cycles > 0U)
            ? static_cast<uint32_t>((static_cast<uint64_t>(avg_cycles) * 1000ULL) / total_avg_cycles)
            : 0U;

        const char *op_name = g_invoke_profiler.event(event_index).tag;
        const tflite::Operator *op = nullptr;
        if (event_index < mapped_event_count) {
            op = flatbuffer_operator_at(event_index);
            if (op != nullptr) {
                op_name = flatbuffer_operator_name(op);
            }
        }

        printk("  op%02d %s avg=%u us share=%u.%u%%",
               event_index,
               op_name != nullptr ? op_name : "UNKNOWN",
               avg_us,
               share_permille / 10U,
               share_permille % 10U);

        if (op != nullptr && op->outputs() != nullptr && op->outputs()->size() > 0) {
            printk(" out=");
            log_flatbuffer_tensor_shape(flatbuffer_tensor_at(op->outputs()->Get(0)));
        }
        printk("\n");
    }
}

void accumulate_invoke_profile()
{
    if (g_invoke_profile_reported) {
        return;
    }

    const int event_count = std::min(g_invoke_profiler.event_count(), kMaxProfiledOps);
    if (event_count <= 0) {
        return;
    }

    for (int event_index = 0; event_index < event_count; ++event_index) {
        g_invoke_profile_cycle_sums[event_index] +=
            g_invoke_profiler.event(event_index).elapsed_cycles;
    }
    ++g_invoke_profile_sample_count;

    if (g_invoke_profile_sample_count >= kInvokeProfileReportAfterInvocations) {
        log_invoke_profile_summary(event_count);
        g_invoke_profile_reported = true;
    }
}

int8_t quantize_to_int8(float value, float scale, int zero_point)
{
    if (scale == 0.0f) {
        return 0;
    }

    int32_t q = static_cast<int32_t>(lroundf(value / scale)) + zero_point;
    if (q < -128) {
        q = -128;
    } else if (q > 127) {
        q = 127;
    }
    return static_cast<int8_t>(q);
}

float dequantize_from_int8(int8_t value, float scale, int zero_point)
{
    return (static_cast<int32_t>(value) - zero_point) * scale;
}

float sigmoidf(float x)
{
    if (x >= 0.0f) {
        const float e = expf(-x);
        return 1.0f / (1.0f + e);
    }

    const float e = expf(x);
    return e / (1.0f + e);
}

void softmax_inplace(float *values, int count)
{
    if (values == nullptr || count <= 0) {
        return;
    }

    float max_value = values[0];
    for (int i = 1; i < count; ++i) {
        if (values[i] > max_value) {
            max_value = values[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {
        values[i] = expf(values[i] - max_value);
        sum += values[i];
    }

    if (sum <= 0.0f) {
        return;
    }

    for (int i = 0; i < count; ++i) {
        values[i] /= sum;
    }
}

Result invalid_result()
{
    Result result = {};
    result.predicted_score = 0.0f;
    result.predicted_label = 0;
    result.output_count = 0;
    for (int i = 0; i < audio_event_model_config::kLabelCount; ++i) {
        result.scores[i] = 0.0f;
    }
    result.inference_ok = false;
    return result;
}

int tensor_element_count(const TfLiteTensor *tensor)
{
    if (tensor == nullptr || tensor->dims == nullptr) {
        return 0;
    }

    int count = 1;
    for (int i = 0; i < tensor->dims->size; ++i) {
        count *= tensor->dims->data[i];
    }
    return count;
}

int tensor_value_count(const TfLiteTensor *tensor)
{
    if (tensor == nullptr) {
        return 0;
    }

    switch (tensor->type) {
    case kTfLiteFloat32:
        return static_cast<int>(tensor->bytes / sizeof(float));
    case kTfLiteInt8:
        return static_cast<int>(tensor->bytes / sizeof(int8_t));
    default:
        return 0;
    }
}

ModelInputSpec infer_model_input_spec(const TfLiteTensor *tensor)
{
    ModelInputSpec spec;
    if (tensor == nullptr || tensor->dims == nullptr) {
        return spec;
    }

    const TfLiteIntArray *dims = tensor->dims;
    spec.element_count = tensor_element_count(tensor);
    if (spec.element_count <= 0) {
        return spec;
    }

    if (dims->size == 4) {
        const int d0 = dims->data[0];
        const int d1 = dims->data[1];
        const int d2 = dims->data[2];
        const int d3 = dims->data[3];

        if (d0 != 1) {
            return spec;
        }
        if (d3 == 1 && d1 > 0 && d2 > 0) {
            spec.layout = InputLayout::kNhwc;
            spec.rows = d1;
            spec.cols = d2;
            spec.channels = d3;
            return spec;
        }
        if (d1 == 1 && d2 > 0 && d3 > 0) {
            spec.layout = InputLayout::kNchw;
            spec.rows = d2;
            spec.cols = d3;
            spec.channels = d1;
            return spec;
        }
        return spec;
    }

    if (dims->size == 3) {
        if (dims->data[0] != 1 || dims->data[1] <= 0 || dims->data[2] <= 0) {
            return spec;
        }
        spec.layout = InputLayout::kMatrix;
        spec.rows = dims->data[1];
        spec.cols = dims->data[2];
        spec.channels = 1;
        return spec;
    }

    if (dims->size == 2) {
        if (dims->data[0] != 1 || dims->data[1] <= 0) {
            return spec;
        }
        spec.layout = InputLayout::kFlat;
        spec.rows = 1;
        spec.cols = dims->data[1];
        spec.channels = 1;
    }

    return spec;
}

bool matches_base_feature_matrix(int rows, int cols)
{
    return rows == kBaseMelBinCount && cols == kBaseFeatureFrameCount;
}

bool matches_transposed_feature_matrix(int rows, int cols)
{
    return rows == kBaseFeatureFrameCount && cols == kBaseMelBinCount;
}

bool matches_flat_feature_vector(int rows, int cols)
{
    return (rows == 1 && cols == kBaseFeatureElementCount) ||
           (rows == kBaseFeatureElementCount && cols == 1);
}

float feature_value_for_input_index(int index)
{
    if (index < 0 || index >= kBaseFeatureElementCount) {
        return 0.0f;
    }

    if (g_input_spec.layout == InputLayout::kFlat) {
        return g_base_feature_buf[index];
    }

    if (matches_base_feature_matrix(g_input_spec.rows, g_input_spec.cols) ||
        matches_flat_feature_vector(g_input_spec.rows, g_input_spec.cols)) {
        return g_base_feature_buf[index];
    }

    if (matches_transposed_feature_matrix(g_input_spec.rows, g_input_spec.cols)) {
        const int frame_idx = index / kBaseMelBinCount;
        const int mel_idx = index % kBaseMelBinCount;
        return g_base_feature_buf[(mel_idx * kBaseFeatureFrameCount) + frame_idx];
    }

    return 0.0f;
}

bool init_frontend_once()
{
    if (g_frontend_initialized) {
        return true;
    }
    if (g_frontend_init_failed) {
        return false;
    }

    const arm_status fft_status = arm_rfft_fast_init_f32(&g_fft_instance, kFftLength);
    if (fft_status != ARM_MATH_SUCCESS) {
        printk("Generated frontend FFT init failed: status=%d\n", static_cast<int>(fft_status));
        g_frontend_init_failed = true;
        return false;
    }

    g_frontend_initialized = true;
    return true;
}

bool extract_base_log_mel_features(const int16_t *samples, int sample_count)
{
    if (samples == nullptr || sample_count < kWindowSampleCount) {
        return false;
    }
    if (!init_frontend_once()) {
        return false;
    }

    std::fill(g_base_feature_buf, g_base_feature_buf + kBaseFeatureElementCount, 0.0f);

    int frames_written = 0;
    for (int start = 0;
         start + kAnalysisFrameSamples <= sample_count &&
         frames_written < kBaseFeatureFrameCount;
         start += kFrameStrideSamples) {
        // 逐帧构造零填充输入，再只对有效窗口区间做缩放和加窗。
        std::fill(g_fft_input, g_fft_input + kFftLength, 0.0f);
        for (int i = 0; i < kFrameFftCopySamples; ++i) {
            const int fft_index = kWindowPadLeft + i;
            g_fft_input[fft_index] = static_cast<float>(samples[start + fft_index]);
        }
        arm_scale_f32(
            g_fft_input + kWindowPadLeft,
            kPcmScale,
            g_fft_input + kWindowPadLeft,
            kFrameFftCopySamples);
        arm_mult_f32(
            g_fft_input + kWindowPadLeft,
            g_window,
            g_fft_input + kWindowPadLeft,
            kFrameFftCopySamples);

        // CMSIS-DSP 的 RFFT 输出先解包成 [re, im] 复数对，再求功率谱。
        arm_rfft_fast_f32(&g_fft_instance, g_fft_input, g_fft_output, 0);
        g_fft_output[kFftLength] = g_fft_output[1];
        g_fft_output[kFftLength + 1] = 0.0f;
        g_fft_output[1] = 0.0f;
        arm_cmplx_mag_squared_f32(g_fft_output, g_power_spectrum, kFftBinCount);

        const float *mel_weights = g_mel_filter_coefs;
        for (int mel_idx = 0; mel_idx < kBaseMelBinCount; ++mel_idx) {
            float mel_energy = 0.0f;
            const uint16_t filter_pos = g_mel_filter_pos[mel_idx];
            const uint16_t filter_len = g_mel_filter_len[mel_idx];
            if (filter_len > 0U) {
                arm_dot_prod_f32(
                    g_power_spectrum + filter_pos,
                    mel_weights,
                    filter_len,
                    &mel_energy);
            }
            mel_weights += filter_len;
            g_base_feature_buf[(mel_idx * kBaseFeatureFrameCount) + frames_written] = mel_energy;
        }

        ++frames_written;
    }

    if (frames_written != kBaseFeatureFrameCount) {
        return false;
    }

    // 先用自然对数向量化，再缩放成 dB，随后按 top_db 做截断。
    arm_offset_f32(
        g_base_feature_buf,
        1.0e-12f,
        g_base_feature_buf,
        kBaseFeatureElementCount);
    arm_vlog_f32(
        g_base_feature_buf,
        g_base_feature_buf,
        kBaseFeatureElementCount);
    arm_scale_f32(
        g_base_feature_buf,
        kNaturalLogToDbScale,
        g_base_feature_buf,
        kBaseFeatureElementCount);

    float max_db = 0.0f;
    uint32_t max_index = 0;
    arm_max_f32(g_base_feature_buf, kBaseFeatureElementCount, &max_db, &max_index);
    const float floor_db = max_db - audio_event_model_config::kTopDb;
    for (int i = 0; i < kBaseFeatureElementCount; ++i) {
        g_base_feature_buf[i] = std::max(g_base_feature_buf[i], floor_db);
    }

    // 这里保持与旧实现一致，使用总体标准差而不是 CMSIS 的 N-1 样本标准差。
    float mean = 0.0f;
    arm_mean_f32(g_base_feature_buf, kBaseFeatureElementCount, &mean);
    arm_offset_f32(
        g_base_feature_buf,
        -mean,
        g_base_feature_buf,
        kBaseFeatureElementCount);

    float centered_energy = 0.0f;
    arm_dot_prod_f32(
        g_base_feature_buf,
        g_base_feature_buf,
        kBaseFeatureElementCount,
        &centered_energy);
    const float variance = std::max(
        centered_energy / static_cast<float>(kBaseFeatureElementCount),
        0.0f);
    float stddev = 0.0f;
    if (arm_sqrt_f32(variance, &stddev) != ARM_MATH_SUCCESS) {
        return false;
    }

    const float normalize_scale = 1.0f / (stddev + 1.0e-6f);
    arm_scale_f32(
        g_base_feature_buf,
        normalize_scale,
        g_base_feature_buf,
        kBaseFeatureElementCount);
    return true;
}

bool validate_model_input_layout()
{
    if (g_input_spec.element_count != kBaseFeatureElementCount) {
        return false;
    }

    if (g_input_spec.layout == InputLayout::kFlat) {
        return true;
    }

    if (g_input_spec.layout == InputLayout::kMatrix) {
        return matches_base_feature_matrix(g_input_spec.rows, g_input_spec.cols) ||
               matches_transposed_feature_matrix(g_input_spec.rows, g_input_spec.cols) ||
               matches_flat_feature_vector(g_input_spec.rows, g_input_spec.cols);
    }

    if ((g_input_spec.layout == InputLayout::kNhwc ||
         g_input_spec.layout == InputLayout::kNchw) &&
        g_input_spec.channels == 1) {
        return matches_base_feature_matrix(g_input_spec.rows, g_input_spec.cols) ||
               matches_transposed_feature_matrix(g_input_spec.rows, g_input_spec.cols) ||
               matches_flat_feature_vector(g_input_spec.rows, g_input_spec.cols);
    }

    return false;
}

bool init_model_once()
{
    if (g_model_initialized) {
        return true;
    }
    if (g_model_init_attempted) {
        printk("Generated model init previously failed: %s\n", g_last_error);
        return false;
    }
    g_model_init_attempted = true;

    if (audio_event_model_data::kPlaceholderModel) {
        set_last_error("placeholder model still active");
        printk("Generated model placeholder is still in use\n");
        return false;
    }

    g_model = tflite::GetModel(audio_event_model_data::g_model);
    if (g_model == nullptr) {
        set_last_error("model pointer is null");
        return false;
    }

    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        set_last_error("schema mismatch");
        printk("Generated model schema mismatch: model=%d runtime=%d\n",
               g_model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    g_model_operator_count = flatbuffer_operator_count();
    if (g_model_operator_count <= 0) {
        set_last_error("model operator list is empty");
        return false;
    }
    if (g_model_operator_count > kMaxProfiledOps) {
        set_last_error("model has more ops than profiler buffer");
        printk("Generated model op count %d exceeds profiler capacity %d\n",
               g_model_operator_count, kMaxProfiledOps);
        return false;
    }
    std::fill(g_invoke_profile_cycle_sums,
              g_invoke_profile_cycle_sums + kMaxProfiledOps,
              0ULL);
    g_invoke_profile_sample_count = 0;
    g_invoke_profile_reported = false;
    g_invoke_profiler.Reset();

    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddAveragePool2D();
    resolver.AddStridedSlice();
    resolver.AddFullyConnected();

    static tflite::MicroInterpreter static_interpreter(
        g_model, resolver, g_tensor_arena, kTensorArenaSize, nullptr, &g_invoke_profiler);
    g_interpreter = &static_interpreter;

    if (g_interpreter->AllocateTensors() != kTfLiteOk) {
        set_last_error("AllocateTensors failed");
        g_interpreter = nullptr;
        return false;
    }

    g_input = g_interpreter->input(0);
    g_output = g_interpreter->output(0);
    if (g_input == nullptr || g_output == nullptr) {
        set_last_error("input/output tensor missing");
        return false;
    }

    g_input_spec = infer_model_input_spec(g_input);
    if (g_input_spec.layout == InputLayout::kUnsupported) {
        set_last_error("unsupported model input shape");
        return false;
    }
    if (g_input_spec.channels != 1) {
        set_last_error("multi-channel model input unsupported");
        return false;
    }
    if (g_input_spec.element_count > audio_event_model_config::kMaxModelInputElementCount) {
        set_last_error("model input exceeds feature buffer capacity");
        return false;
    }

    g_output_count = tensor_value_count(g_output);
    if (g_output_count <= 0) {
        set_last_error("unsupported output tensor type");
        return false;
    }

    g_model_initialized = true;
    set_last_error("ok");
    printk("Generated model ready: input_elements=%d output_count=%d arena_used=%u B arena_reserved=%u B\n",
           g_input_spec.element_count,
           g_output_count,
           static_cast<unsigned int>(g_interpreter->arena_used_bytes()),
           static_cast<unsigned int>(kTensorArenaSize));
    printk("Invoke profiler armed: average first %u runs, then print per-op breakdown once\n",
           static_cast<unsigned int>(kInvokeProfileReportAfterInvocations));
    return true;
}

}  // namespace

bool model_is_placeholder()
{
    return audio_event_model_data::kPlaceholderModel;
}

unsigned int runtime_memory_size_bytes()
{
    return static_cast<unsigned int>(
        sizeof(g_base_feature_buf) + sizeof(g_tensor_arena));
}

const char *last_error()
{
    return g_last_error;
}

uint32_t last_feature_ms()
{
    return g_last_feature_ms;
}

uint32_t last_invoke_ms()
{
    return g_last_invoke_ms;
}

Result run_classifier(const int16_t *samples, int sample_count)
{
    g_last_feature_ms = 0;
    g_last_invoke_ms = 0;

    if (samples == nullptr || sample_count != kWindowSampleCount) {
        set_last_error("invalid audio window");
        return invalid_result();
    }
    if (!init_model_once()) {
        return invalid_result();
    }

    const uint32_t feature_start_ms = k_uptime_get_32();
    if (!extract_base_log_mel_features(samples, sample_count)) {
        set_last_error("frontend extraction failed");
        return invalid_result();
    }
    if (!validate_model_input_layout()) {
        set_last_error("model input layout mismatch");
        return invalid_result();
    }
    g_last_feature_ms = k_uptime_get_32() - feature_start_ms;

    if (g_input->type == kTfLiteInt8) {
        for (int i = 0; i < g_input_spec.element_count; ++i) {
            const float feature_value = feature_value_for_input_index(i);
            g_input->data.int8[i] = quantize_to_int8(
                feature_value, g_input->params.scale, g_input->params.zero_point);
        }
    } else if (g_input->type == kTfLiteFloat32) {
        for (int i = 0; i < g_input_spec.element_count; ++i) {
            g_input->data.f[i] = feature_value_for_input_index(i);
        }
    } else {
        set_last_error("unsupported input tensor type");
        return invalid_result();
    }

    g_invoke_profiler.Reset();
    const uint32_t invoke_start_ms = k_uptime_get_32();
    if (g_interpreter->Invoke() != kTfLiteOk) {
        set_last_error("Invoke failed");
        return invalid_result();
    }
    g_last_invoke_ms = k_uptime_get_32() - invoke_start_ms;
    accumulate_invoke_profile();

    Result result = {};
    result.output_count = g_output_count;
    if (g_output->type == kTfLiteInt8) {
        for (int i = 0; i < g_output_count; ++i) {
            const float value = dequantize_from_int8(
                g_output->data.int8[i], g_output->params.scale, g_output->params.zero_point);
            result.scores[i] = value;
        }
    } else if (g_output->type == kTfLiteFloat32) {
        for (int i = 0; i < g_output_count; ++i) {
            result.scores[i] = g_output->data.f[i];
        }
    } else {
        set_last_error("unsupported output tensor type");
        return invalid_result();
    }

    if (g_output_count == 1) {
        result.scores[0] = sigmoidf(result.scores[0]);
        result.predicted_score = result.scores[0];
        result.predicted_label = (result.predicted_score >= kDefaultDecisionThreshold) ? 1 : 0;
    } else {
        softmax_inplace(result.scores, g_output_count);
        result.predicted_label = 0;
        result.predicted_score = result.scores[0];
        for (int i = 1; i < g_output_count; ++i) {
            if (result.scores[i] > result.predicted_score) {
                result.predicted_score = result.scores[i];
                result.predicted_label = i;
            }
        }
    }

    set_last_error("ok");
    result.inference_ok = true;
    return result;
}

}  // namespace audio_event_model_runner
