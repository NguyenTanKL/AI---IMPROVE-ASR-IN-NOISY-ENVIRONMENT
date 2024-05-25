import pathlib
import os
from pydub import AudioSegment
#Chọn file để chuyển sang định dạng PCM
file_path = "SpeechCommands_Musan\\speech_command_test_musan_40_dB"
def convert_wav_to_pcm(input_file):
    # Đọc tệp WAV
    audio = AudioSegment.from_wav(input_file)
    
    # Chuyển đổi sang PCM (16-bit, mono, tần số mẫu 16kHz)
    audio = audio.set_frame_rate(16000)  # Cài đặt tần số lấy mẫu là 16kHz
    audio = audio.set_channels(1)        # Cài đặt số kênh âm thanh là mono
    audio = audio.set_sample_width(2)    # Số byte mỗi mẫu (PCM 16-bit)
    
    # Ghi đè tệp WAV ban đầu bằng dữ liệu PCM đã chuyển đổi
    audio.export(input_file, format='wav')

for root, folders, files in os.walk(file_path):
    # Duyệt qua tất cả các tệp trong mỗi thư mục con
    for file in files:
        # Chỉ xử lý các tệp có phần mở rộng là .wav
        if file.endswith(".wav"):
            # Tạo đường dẫn đầy đủ của tệp đầu vào
            input_path = os.path.join(root, file)
            # Thực hiện chuyển đổi từ WAV sang PCM
            convert_wav_to_pcm(input_path)