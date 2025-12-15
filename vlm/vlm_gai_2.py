import numpy as np
import torch
import json
import re
import base64
import textwrap
import queue
import time
import io
import os
import cv2
import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment
from ultralytics.models.sam import Predictor as SAMPredictor
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoProcessor
from PIL import Image
import websocket
import hashlib
import hmac
from urllib.parse import urlencode
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread

# 讯飞语音识别参数
STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识
global recognition_result  # 用于存储识别结果
recognition_result = ""


class Ws_Param(object):
    def __init__(self, APPID, APIKey, APISecret):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.CommonArgs = {"app_id": self.APPID}
        self.BusinessArgs = {"domain": "iat", "language": "zh_cn", "accent": "mandarin", "vinfo":1, "vad_eos":10000}

    def create_url(self):
        url = 'wss://ws-api.xfyun.cn/v2/iat'
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/iat " + "HTTP/1.1"
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f"api_key=\"{self.APIKey}\", algorithm=\"hmac-sha256\", headers=\"host date request-line\", signature=\"{signature_sha}\""
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        return url + '?' + urlencode(v)


def on_message(ws, message):
    global recognition_result
    try:
        res = json.loads(message)
        if res.get("code") != 0:
            errMsg = res.get("message", "")
            print(f"识别错误: {errMsg} (code: {res.get('code')})")
            return

        data = res.get("data", {}).get("result", {}).get("ws", [])
        current_result = ""
        for item in data:
            for cw in item.get("cw", []):
                current_result += cw.get("w", "")
        
        # 过滤无效结果
        if current_result not in ['。', '.。', ' .。', ' 。']:
            recognition_result = current_result
            print(f"实时识别结果: {recognition_result}")
    except Exception as e:
        print(f"消息解析异常: {e}")


def on_error(ws, error):
    print(f"WebSocket错误: {error}")


def on_close(ws, close_status_code=None, close_msg=None):
    print("WebSocket连接已关闭")
    if close_status_code:
        print(f"关闭状态码: {close_status_code}")
    if close_msg:
        print(f"关闭消息: {close_msg}")


def on_open(ws):
    def run(*args):
        status = STATUS_FIRST_FRAME
        CHUNK = 520
        FORMAT = 'int16'
        CHANNELS = 1
        RATE = 16000
        
        with sd.RawInputStream(samplerate=RATE, blocksize=CHUNK,
                              dtype=FORMAT, channels=CHANNELS) as stream:
            print("🎤 开始录音...")
            silence_threshold = 300
            silence_counter = 0
            max_silence_chunks = int(2.0 * RATE / CHUNK)  # 2秒静音停止

            while True:
                buf, overflowed = stream.read(CHUNK)
                if overflowed:
                    print("警告: 音频缓冲区溢出")

                # 修正：先转 float32 再平方，避免 int16 溢出
                samples = np.frombuffer(buf, dtype=np.int16)
                volume = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) if samples.size else 0.0
                
                if volume < silence_threshold:
                    silence_counter += 1
                    if silence_counter > max_silence_chunks and status != STATUS_FIRST_FRAME:
                        status = STATUS_LAST_FRAME
                else:
                    silence_counter = 0
                
                try:
                    if status == STATUS_FIRST_FRAME:
                        d = {
                            "common": ws.param.CommonArgs,
                            "business": ws.param.BusinessArgs,
                            "data": {
                                "status": 0,
                                "format": "audio/L16;rate=16000",
                                "audio": base64.b64encode(buf).decode('utf-8'),
                                "encoding": "raw"
                            }
                        }
                        ws.send(json.dumps(d))
                        status = STATUS_CONTINUE_FRAME
                    elif status == STATUS_CONTINUE_FRAME:
                        d = {
                            "data": {
                                "status": 1,
                                "format": "audio/L16;rate=16000",
                                "audio": base64.b64encode(buf).decode('utf-8'),
                                "encoding": "raw"
                            }
                        }
                        ws.send(json.dumps(d))
                    elif status == STATUS_LAST_FRAME:
                        d = {
                            "data": {
                                "status": 2,
                                "format": "audio/L16;rate=16000",
                                "audio": base64.b64encode(buf).decode('utf-8'),
                                "encoding": "raw"
                            }
                        }
                        ws.send(json.dumps(d))
                        time.sleep(0.5)
                        break
                except Exception as e:
                    print(f"发送音频数据失败: {e}")
                    break
        
        ws.close()
        print("🛑 录音结束")
    
    thread.start_new_thread(run, ())


def recognize_speech():
    """使用讯飞开放平台WebSocket进行语音识别"""
    global recognition_result
    recognition_result = ""
    
    # 请替换为实际的讯飞API账号信息
    APPID = "ccfb1298"
    APIKey = "badc41fa2c23028b4d110e47a89d6da4"
    APISecret = "YjZhMzk0YzkyNDQxMDc1ZDY1Yzk0NjNi"
    
    wsParam = Ws_Param(APPID, APIKey, APISecret)
    wsUrl = wsParam.create_url()
    
    ws = websocket.WebSocketApp(wsUrl,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    ws.param = wsParam  # 附加参数供on_open使用
    ws.on_open = on_open
    
    print("🎙️ 启动语音识别，请说话...")
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE}, ping_timeout=2)
    
    return recognition_result


# ----------------------- 基础工具函数 -----------------------
def encode_np_array(image_np):
    """将 numpy 图像数组（BGR）编码为 base64 字符串"""
    success, buffer = cv2.imencode('.jpg', image_np)
    if not success:
        raise ValueError("无法将图像数组编码为 JPEG")
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64


def plot_coordinates(image_input, coords):
    """根据坐标打印简单提示，但不显示图像"""
    if not coords:
        print("[提示] 未提取到坐标信息。")
        return
    if "bbox" in coords:
        print(f"[标注] 边界框坐标: {coords['bbox']}")
    elif "point" in coords:
        print(f"[标注] 中心点坐标: {coords['point']}")
    elif "x" in coords and "y" in coords:
        print(f"[标注] 中心点坐标: ({coords['x']}, {coords['y']})")


# ----------------------- 多模态模型调用（transformers 本地加载） -----------------------
def generate_robot_actions(user_command, image_input=None):
    """
    使用 Qwen2.5-VL 多模态模型处理用户指令和图像
    返回包含 "response" 和 "coordinates" 的字典
    """
    model_path = r"D:\studentcreate\qwen"  # 原始字符串避免转义

    # 1. 加载 Processor（Qwen2.5-VL 正确做法，包含 tokenizer + image_processor）
    try:
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
    except Exception as e:
        import traceback
        with open("processor_err.txt", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print(f"❌ Processor 加载失败：{e}（详情见 processor_err.txt）")
        return {"response": "Processor 加载失败", "coordinates": {}}

    # 2. 加载模型
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        use_cuda = torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32
        device_map = "auto" if use_cuda else None
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            load_in_8bit=False,
            local_files_only=True
        )
        model.eval()
    except Exception as e:
        import traceback
        with open("model_load_err.txt", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print(f"❌ 模型加载失败：{e}（详情见 model_load_err.txt）")
        return {"response": "模型加载失败", "coordinates": {}}

    # 3. 构建增强版系统提示（保留原逻辑，扩充细节描述）
    system_prompt = textwrap.dedent("""\
    你是一个为精密机械臂设计的视觉控制系统，具备先进的多模态感知能力和空间推理能力。请严格按照以下步骤执行任务，确保输出的准确性和可解析性：

    【图像分析阶段】
    1. 仔细分析输入的彩色图像，识别图像中所有可见物体（包括但不限于几何形状、颜色、纹理、相对位置等视觉特征）。
    2. 为每个识别的物体记录以下信息：
       - 类别名称（如"白色长方体"、"红色圆柱"、"黑色键盘"等）
       - 边界框坐标（格式：[左上角x, 左上角y, 右下角x, 右下角y]，单位为像素，坐标原点为图像左上角）
       - 物体在图像中的相对位置描述（如"位于图像中心偏左"、"靠近画面右上角"等）

    【指令解析与匹配阶段】
    3. 理解用户的自然语言指令，提取关键信息：
       - 目标物体的视觉特征（颜色、形状、大小等）
       - 位置限定词（如"中间的"、"最左边的"、"靠近某物体的"等）
       - 操作意图（如"识别"、"分割"、"抓取"等）
    4. 根据提取的关键信息，从已识别的所有物体中筛选出**唯一**匹配的目标物体：
       - 若有多个物体符合描述，优先选择与位置限定词最匹配的物体
       - 若指令模糊，选择视觉特征最显著、最易于操作的物体
       - 若无匹配物体,在自然语言响应中说明原因（如"图像中未找到符合描述的物体"）

    【响应生成阶段】
    5. 输出格式必须严格遵循以下两部分结构：
       **第一部分：自然语言响应**
       - 用简洁、友好的语言说明为什么选择该物体（包括物体的关键特征和位置信息）
       - 可适当使用俏皮、可爱的语气，但必须清晰传达被选中的物体及其原因
       - 若未找到匹配物体，明确说明原因并给出建议（如"图像中有3个白色物体，但都不在中间位置，建议重新描述"）
       - **注意**：此部分只描述被选中的目标物体，不要列举其他物体

       **第二部分：结构化 JSON 数据**
       - 从下一行开始（与自然语言响应之间无其他文本），返回标准 JSON 对象
       - JSON 格式如下：
       ```json
       {
         "name": "物体的完整类别名称（如'中间的白色长方体'）",
         "bbox": [左上角x, 左上角y, 右下角x, 右下角y]
       }
       ```
       - 若未找到匹配物体，返回空的 bbox（如 `"bbox": []`）

    【输出约束与质量保证】
    6. JSON 对象必须满足以下所有要求：
       - 独立成行，与自然语言响应之间用换行符分隔
       - 不能包含注释、额外文本、解释性说明或代码块标记（如 ```json）
       - 所有坐标值必须为整数（若模型输出浮点数，需四舍五入）
       - 坐标值必须在图像有效范围内（0 ≤ x < 图像宽度, 0 ≤ y < 图像高度）
       - 边界框必须合法（右下角坐标 > 左上角坐标）
       - **只允许**使用 "bbox" 键表示边界框，禁止使用 "box"、"bounding_box" 等其他变体

    【示例输出格式】
    好的！我识别到了图像中间的白色长方体，它位于画面的正中央偏左，尺寸约为 200x150 像素，是最显眼的目标物体哦！

    {
      "name": "中间的白色长方体",
      "bbox": [450, 280, 650, 430]
    }

    【错误示例（禁止）】
    ❌ 错误1：JSON 与文本混在一起
    我找到了白色长方体 {"name": "...", "bbox": [...]}

    ❌ 错误2：包含代码块标记
    ```json
    {"name": "...", "bbox": [...]}
    ```

    ❌ 错误3：使用非法键名
    {"name": "...", "box": [...]}  // 应为 "bbox"

    ❌ 错误4：坐标为浮点数或字符串
    {"name": "...", "bbox": [100.5, 200.3, "300", "400"]}  // 必须为整数

    【最后提醒】
    - 请严格遵守上述所有要求，确保输出的 JSON 可被 Python 的 json.loads() 解析
    - 若有任何不确定的地方，优先保证 JSON 格式正确（可在自然语言部分说明不确定性）
    """)

    # 4. 构建消息列表（system + user with image）
    messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
    user_content = []
    image_pil = None
    if image_input is not None:
        image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        user_content.append({"type": "image", "image": image_pil})
    user_content.append({"type": "text", "text": user_command})
    messages.append({"role": "user", "content": user_content})

    # 5. 用 processor.apply_chat_template 格式化并推理（正确的 Qwen2.5-VL 流程）
    try:
        chat_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[chat_text],
            images=[image_pil] if image_pil is not None else None,
            return_tensors="pt"
        )
        # 把张量移到模型设备
        inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        # ✅ 关键修复：只解码新生成的 token（跳过输入部分），避免包含 system/user prompt
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        content = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # ✅ 统一替换全角引号为半角，避免 JSON 解析失败
        content = content.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
        
        print("原始响应：", content)
    except Exception as e:
        import traceback
        with open("inference_err.txt", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print(f"模型推理失败：{e}（详情见 inference_err.txt）")
        return {"response": "处理失败", "coordinates": {}}

    # 6. 解析响应（提取自然语言与 JSON）—— 用更健壮的正则
    # 匹配包含 "bbox" 的 JSON 对象（支持多行、允许前后有空格/换行）
    match = re.search(r'\{[^{}]*"bbox"[^{}]*\}', content, re.DOTALL)
    if match:
        json_str = match.group(0).strip()
        try:
            coord = json.loads(json_str)
            if "bbox" in coord and coord["bbox"]:
                # 确保坐标为整数
                coord["bbox"] = [int(round(float(x))) for x in coord["bbox"]]
            else:
                coord = {}
        except Exception as e:
            print(f"[警告] JSON 解析失败：{e}")
            print(f"[调试] 提取到的 JSON 字符串：{repr(json_str)}")
            coord = {}
        # 自然语言部分 = JSON 之前的内容
        natural_response = content[:match.start()].strip()
    else:
        natural_response = content.strip()
        coord = {}
        print("[警告] 未提取到有效的 JSON 数据")

    return {"response": natural_response, "coordinates": coord}


# ----------------------- SAM 分割相关 -----------------------
def choose_model():
    """初始化 SAM 分割预测器，设置相关参数"""
    model_weight = 'sam_b.pt'  # 若未下载，Ultralytics 会自动下载
    overrides = dict(
        task='segment',
        mode='predict',
        model=model_weight,
        conf=0.01,
        save=False
    )
    return SAMPredictor(overrides=overrides)


def process_sam_results(results):
    """处理 SAM 分割结果，获取掩码和中心点"""
    if not results or not results[0].masks:
        return None, None
    mask = results[0].masks.data[0].cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    M = cv2.moments(contours[0])
    if M["m00"] == 0:
        return None, mask
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask


# ----------------------- 语音识别与 TTS（替换为讯飞WebSocket实现） -----------------------
# 语音录制参数保持不变（用于音频采集配置）
samplerate = 16000  # 讯飞也使用16000采样率
channels = 1
dtype = 'int16'
frame_duration = 0.2
frame_samples = int(frame_duration * samplerate)
silence_threshold = 500  # 静音阈值，可根据环境调整
silence_max_duration = 1.0  # 静音超过 1 秒停止录音


def speech_to_text(audio_data=None):
    """
    适配原接口，调用讯飞WebSocket语音识别
    audio_data参数保留但不使用（兼容原函数调用方式）
    """
    print("📡 正在使用讯飞开放平台识别语音...")
    text = recognize_speech()
    if text:
        print(f"✅ 语音识别结果：{text}")
        return text
    else:
        print("❌ 未识别到有效文本")
        return ""


def play_tts(text):
    """
    调用 TTS 接口（若没有 TTS 服务，可注释此函数内部逻辑，仅打印文本）
    若有本地 TTS 服务，修改 URL 为实际地址；若无，可替换为 pyttsx3 本地播报
    """
    # 方案2：若无 TTS 服务，用 pyttsx3 本地播报（需先安装：pip install pyttsx3）
    import pyttsx3
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    print("📢 TTS 播报：", text)
    return

    # 以下为原注释的TTS方案，保持不变
    # print("📢 准备播报 TTS 语音：", text)
    # payload = {
    #     "model": "cosyvoice",
    #     "voice": "tarzan",
    #     "input": text,
    #     "response_format": "mp3",
    #     "speed": 1.2
    # }
    # headers = {
    #     "Authorization": f"Bearer {TTS_API_TOKEN}",
    #     "Content-Type": "application/json"
    # }
    # try:
    #     response = requests.post(TTS_API_URL, headers=headers, data=json.dumps(payload), timeout=10)
    #     if response.ok:
    #         audio_data = response.content
    #         audio_io = io.BytesIO(audio_data)
    #         audio_seg = AudioSegment.from_file(audio_io, format="mp3")
    #         raw = np.array(audio_seg.get_array_of_samples())
    #         raw = raw.reshape((-1, audio_seg.channels))
    #         sd.play(raw, audio_seg.frame_rate)
    #         sd.wait()
    #     else:
    #         print(f"❌ TTS 接口失败：状态码 {response.status_code}")
    #         #  fallback：若无 TTS 服务，用文本提示
    #         print(f"📢 （备用）播报内容：{text}")
    # except Exception as e:
    #     print(f"❌ TTS 播报失败：{e}")
    #     print(f"📢 （备用）播报内容：{text}")


def voice_command_to_keyword():
    """获取语音命令并转换为文本（保持原接口不变）"""
    # 直接调用speech_to_text，内部已实现讯飞识别
    text = speech_to_text()
    if not text:
        print("⚠️ 没有识别到文本指令")
        return ""
    print(f"✅ 最终指令：{text}")
    return text


# ----------------------- 主流程：图像分割 -----------------------
def segment_image(image_input, output_mask='mask.png'):
    """
    自动语音获取检测目标 → 多模态模型检测 → SAM 分割 → 保存掩码
    参数 image_input 为 numpy 数组（BGR 格式）。
    """
    # 1. 使用语音获取目标指令
    print("🎙️ 请通过语音描述目标物体及抓取指令（例如：'识别并分割红色杯子'）...")
    command_text = voice_command_to_keyword()
    if not command_text:
        print("⚠️ 未获取到有效指令，终止流程。")
        return None
    print(f"✅ 确认指令：{command_text}")

    # 2. 通过多模态模型获取检测框
    print("🔍 多模态模型正在分析图像和指令...")
    result = generate_robot_actions(command_text, image_input)
    natural_response = result["response"]
    detection_info = result["coordinates"]
    print("\n🗣️ 模型回应：", natural_response)
    print("📊 检测结果：", detection_info)

    # 3. TTS 播报模型回应（若无 TTS 服务，仅打印文本）
    play_tts(natural_response)

    # 4. 提取边界框（若未检测到，提示手动选择）
    bbox = detection_info.get("bbox") if (detection_info and "bbox" in detection_info) else None
    if bbox:
        # 验证 bbox 有效性（避免坐标超出图像范围）
        img_h, img_w = image_input.shape[:2]
        bbox = [max(0, x) for x in bbox]
        bbox[2] = min(img_w, bbox[2])
        bbox[3] = min(img_h, bbox[3])
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            print("⚠️ 检测到的边界框无效，将手动选择目标。")
            bbox = None

    # 5. 初始化 SAM 并处理图像
    print("\n✂️ 准备进行图像分割...")
    image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)  # SAM 需 RGB 格式
    predictor = choose_model()
    predictor.set_image(image_rgb)  # 设置 SAM 输入图像

    # 6. 分割目标（自动检测或手动选择）
    if bbox:
        print(f"✅ 使用自动检测的边界框：{bbox}")
        results = predictor(bboxes=[bbox])  # SAM 基于边界框分割
    else:
        print("⚠️ 未检测到有效边界框，请点击图像中的目标区域（左键单击）...")
        cv2.namedWindow('Select Object', cv2.WINDOW_NORMAL)
        cv2.imshow('Select Object', image_input)
        cv2.resizeWindow('Select Object', 800, 600)  # 调整窗口大小
        point = []

        # 鼠标点击回调函数（获取点击坐标）
        def click_handler(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                point.extend([x, y])
                print(f"🖱️ 已选择坐标：({x}, {y})")
                cv2.setMouseCallback('Select Object', lambda *args: None)  # 禁用后续点击

        cv2.setMouseCallback('Select Object', click_handler)
        # 等待用户点击或关闭窗口
        while True:
            key = cv2.waitKey(100)
            if point:  # 检测到点击
                break
            if cv2.getWindowProperty('Select Object', cv2.WND_PROP_VISIBLE) < 1:  # 窗口被关闭
                print("❌ 窗口已关闭，终止分割流程。")
                cv2.destroyAllWindows()
                return None
        cv2.destroyAllWindows()
        print(f"✅ 使用手动选择的坐标：{point}")
        results = predictor(points=[point], labels=[1])  # SAM 基于点分割（label=1 表示前景）

    # 7. 处理分割结果并保存掩码
    center, mask = process_sam_results(results)
    if mask is not None:
        output_dir = r"D:\studentcreate\graspnet-baseline\doc\example_data"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_mask)
        cv2.imwrite(output_path, mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
        print(f"\n✅ 分割完成！掩码已保存为：{output_path}")
        print(f"🎯 目标中心点坐标：{center}（若有）")
    else:
        print("\n⚠️ 分割失败，未生成有效掩码。")
    return mask

if __name__ == "__main__":
    import traceback
    import sys
    try:
        # 1. 读取输入图像（替换为你的图像路径）
        input_image_path = "color.png"  # 图像路径（相对或绝对路径）
        input_image = cv2.imread(input_image_path)
        
        # 2. 验证图像是否读取成功
        if input_image is None:
            raise ValueError(f"❌ 无法读取图像文件：{input_image_path}\n请检查路径是否正确，或图像文件是否损坏。")
        print(f"✅ 成功读取图像：{input_image_path}（尺寸：{input_image.shape[1]}x{input_image.shape[0]}）")

        # 可选：用本地音频文件替代麦克风识别（取消注释即可启用）
        '''
        use_local_audio = True
        audio_file_path = "test_audio.wav"  # 本地音频文件路径
        if use_local_audio:
            from pydub import AudioSegment
            def transcribe_local_audio():
                audio = AudioSegment.from_file(audio_file_path)
                samples = np.array(audio.get_array_of_samples())
                if audio.channels > 1:
                    samples = samples.reshape((-1, audio.channels))[:, 0]
                return speech_to_text(samples)
            
            # 替换语音识别函数
            global voice_command_to_keyword
            voice_command_to_keyword = transcribe_local_audio
            print(f"🔊 使用本地音频文件：{audio_file_path}")
        else:
            print("🎙️ 使用麦克风实时识别")
        '''

        # 3. 执行核心分割流程
        seg_mask = segment_image(input_image)

        # 4. 输出最终结果状态
        if seg_mask is not None:
            print(f"\n🎉 流程结束！分割掩码尺寸：{seg_mask.shape}")
        else:
            print("\n❌ 流程结束，分割未成功。")

    except Exception as e:
        tb = traceback.format_exc()
        print("❌ 脚本运行出现异常，详情已保存到 err_trace.txt")
        with open("err_trace.txt", "w", encoding="utf-8") as f:
            f.write(tb)
        sys.stderr.write(tb)
        # 防止程序立即退出（供调试查看）
        try:
            input("按回车退出...")
        except:
            pass