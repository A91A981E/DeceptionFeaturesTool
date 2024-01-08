import os
import wave
from pydub import AudioSegment
import contextlib
import subprocess
import csv
from tqdm import tqdm


class AudioProcessor:
    def __init__(
        self,
        audio_path,
        path_segment="cache",
        opensmile_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resources", "SMILExtract.exe"
        ),
        config=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resources", "emo_large.conf"
        ),
        output_folder=os.path.join("output", "OpenSMILE"),
        window_size=100,
    ):
        if not os.path.exists(path_segment):
            os.makedirs(path_segment)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # Initialize class variables
        self.audio_path = audio_path
        self.path_segment = path_segment
        self.opensmile_path = opensmile_path
        self.config = config
        self.output_folder = output_folder
        self.window_size = window_size

        # Start audio processing
        self.divide(self.window_size)
        self.extract_features()

    def get_wav_time(self, wav_path):
        """Get duration of the wav file"""
        with contextlib.closing(wave.open(wav_path, "r")) as f:
            frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration

    def get_ms_part_wav(self, main_wav_path, start_time, end_time, part_wav_path):
        """Extract a specific part of the audio file based on start and end times"""
        start_time = int(start_time)
        end_time = int(end_time)
        sound = AudioSegment.from_mp3(main_wav_path)
        word = sound[start_time:end_time]
        word.export(part_wav_path, format="wav")

    def divide(self, time_segment):
        """Divide the audio into multiple segments"""
        for root, dir, files in os.walk(self.audio_path):
            for i in tqdm(range(len(files))):
                if os.path.splitext(files[i])[1] == ".wav":
                    audio = os.path.join(root, files[i])
                    time_all = int(self.get_wav_time(audio) * 1000)
                    start_time = 0
                    index = 1
                    while start_time <= time_all - time_segment:
                        end_time = start_time + time_segment
                        aduio_segment = os.path.join(
                            self.path_segment, f"{files[i][:-4]}_{'%04d' % index}.wav"
                        )
                        self.get_ms_part_wav(audio, start_time, end_time, aduio_segment)
                        start_time += time_segment
                        index += 1
                    aduio_segment = os.path.join(
                        self.path_segment, f"{files[i][:-4]}_{'%04d' % index}.wav"
                    )
                    self.get_ms_part_wav(audio, start_time, time_all, aduio_segment)

    def extract_feature(self, audio_file, i):
        """使用 opensmile 提取音频特征"""
        try:
            cmd = f'"{self.opensmile_path}" -C "{self.config}" -I "{os.path.join(audio_file, i)}" -instname "{i}" -csvoutput "{self.output_folder}/{i[:-15]}_opensmile.csv" -noconsoleoutput 1 -appendcsv 1'
            subprocess.run(cmd, shell=True)
        except Exception as e:
            with open(os.path.join(self.output_folder, "exception.log"), "a") as f:
                f.write(f"Error processing file {i}: {str(e)}\n\n")

    def extract_features(self):
        """为所有音频片段提取特征"""
        for root, dir, files in os.walk(self.path_segment):
            for i in tqdm(files):
                self.extract_feature(self.path_segment, i)


if __name__ == "__main__":
    audio_path = r"D:\Postgraduate\Motion\Deception Detection\RealLifeDeceptionDetection.2016\Real-life_Deception_Detection_2016\Audios"
    processor = AudioProcessor(audio_path)
    print("Done")
