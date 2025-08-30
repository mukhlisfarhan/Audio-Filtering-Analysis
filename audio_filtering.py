import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import butter, lfilter

class AudioFilterApp:
    def __init__(self, master):
        self.master = master
        master.title("Aplikasi Filtering Audio")
        master.geometry("1000x800")
        
        self.audio_data = None
        self.sample_rate = None
        self.filtered_data = None
        
        # Atur validasi untuk input numerik
        self.vcmd = (master.register(self.validate_numeric_input), '%P')
        
        self.create_widgets()

    def validate_numeric_input(self, P):
        """Validasi untuk memastikan input hanya berisi angka."""
        if P.strip() == "":
            return True
        try:
            float(P)
            return True
        except ValueError:
            return False

    def create_widgets(self):
        # Frame untuk kontrol
        control_frame = tk.Frame(self.master, padx=10, pady=10)
        control_frame.pack(fill="x")
        
        self.load_button = tk.Button(control_frame, text="Unggah File Audio", command=self.load_audio)
        self.load_button.pack(side="left", padx=5)
        
        self.filter_type_label = tk.Label(control_frame, text="Pilih Filter:")
        self.filter_type_label.pack(side="left", padx=5)
        
        self.filter_type = tk.StringVar(self.master)
        self.filter_type.set("low-pass")
        self.filter_options = ["low-pass", "high-pass", "band-pass"]
        self.filter_menu = tk.OptionMenu(control_frame, self.filter_type, *self.filter_options)
        self.filter_menu.pack(side="left", padx=5)
        
        self.cutoff_label = tk.Label(control_frame, text="Frekuensi Cutoff Bawah (Hz):")
        self.cutoff_label.pack(side="left", padx=5)
        
        self.cutoff_entry = tk.Entry(control_frame, width=10, validate='key', validatecommand=self.vcmd)
        self.cutoff_entry.insert(0, "500")
        self.cutoff_entry.pack(side="left", padx=5)

        self.upper_cutoff_label = tk.Label(control_frame, text="Frekuensi Cutoff Atas (Hz):")
        self.upper_cutoff_label.pack(side="left", padx=5)
        self.upper_cutoff_entry = tk.Entry(control_frame, width=10, validate='key', validatecommand=self.vcmd)
        self.upper_cutoff_entry.insert(0, "2000")
        self.upper_cutoff_entry.pack(side="left", padx=5)
        
        self.apply_button = tk.Button(control_frame, text="Terapkan Filter", command=self.apply_filter)
        self.apply_button.pack(side="left", padx=5)

        self.filter_type.trace_add("write", self.update_cutoff_inputs)
        self.update_cutoff_inputs()
        
        # Frame untuk visualisasi
        self.plot_frame = tk.Frame(self.master, padx=10, pady=10)
        self.plot_frame.pack(fill="both", expand=True)
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)
        
    def load_audio(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio files", "*.wav *.flac *.ogg")]
        )
        if file_path:
            try:
                self.audio_data, self.sample_rate = sf.read(file_path)
                messagebox.showinfo("Sukses", "File audio berhasil diunggah!")
                self.plot_audio()
            except Exception as e:
                messagebox.showerror("Error", f"Gagal memuat file: {e}")

    def butter_filter(self, data, cutoff, fs, btype='low', order=5):
        """
        Menerapkan filter Butterworth pada data audio.
        cutoff dapat berupa nilai tunggal (float) untuk low/high-pass
        atau list/array [lower_cutoff, upper_cutoff] untuk band-pass.
        """
        nyq = 0.5 * fs
        
        if isinstance(cutoff, (list, np.ndarray)):
            # Jika cutoff adalah list/array (untuk band-pass)
            normal_cutoff = [c / nyq for c in cutoff]
        else:
            # Jika cutoff adalah nilai tunggal (untuk low/high-pass)
            normal_cutoff = cutoff / nyq
            
        b, a = butter(order, normal_cutoff, btype=btype, analog=False)
        y = lfilter(b, a, data)
        return y

    def apply_filter(self):
        if self.audio_data is None:
            messagebox.showerror("Error", "Mohon unggah file audio terlebih dahulu.")
            return

        try:
            filter_type = self.filter_type.get()
            print(f"Filter yang dipilih: {filter_type}")
            
            # Map the filter type to what scipy.signal.butter expects
            if filter_type == "low-pass":
                scipy_btype = "low"
            elif filter_type == "high-pass":
                scipy_btype = "high"
            elif filter_type == "band-pass":
                scipy_btype = "bandpass"
            else:
                raise ValueError("Jenis filter tidak valid.")
            
            print(f"Jenis filter untuk scipy: {scipy_btype}")

            if filter_type == "band-pass":
                lower_cutoff_str = self.cutoff_entry.get().strip()
                upper_cutoff_str = self.upper_cutoff_entry.get().strip()

                print(f"Input Frekuensi Bawah (string): '{lower_cutoff_str}'")
                print(f"Input Frekuensi Atas (string): '{upper_cutoff_str}'")

                if not lower_cutoff_str or not upper_cutoff_str:
                    messagebox.showerror("Error", "Frekuensi cutoff tidak boleh kosong.")
                    return

                lower_cutoff = float(lower_cutoff_str)
                upper_cutoff = float(upper_cutoff_str)
                
                print(f"Frekuensi Bawah (float): {lower_cutoff}")
                print(f"Frekuensi Atas (float): {upper_cutoff}")

                if lower_cutoff >= upper_cutoff:
                    messagebox.showerror("Error", "Frekuensi cutoff bawah harus lebih kecil dari frekuensi cutoff atas.")
                    return
                cutoff_freq = [lower_cutoff, upper_cutoff]
            else:
                cutoff_str = self.cutoff_entry.get().strip()
                print(f"Input Frekuensi (string): '{cutoff_str}'")

                if not cutoff_str:
                    messagebox.showerror("Error", "Frekuensi cutoff tidak boleh kosong.")
                    return
                cutoff_freq = float(cutoff_str)
                print(f"Frekuensi (float): {cutoff_freq}")
            
            self.filtered_data = self.butter_filter(self.audio_data, cutoff_freq, self.sample_rate, btype=scipy_btype)
            print("Filter berhasil diterapkan!")
            self.plot_audio()
        except ValueError:
            messagebox.showerror("Error", "Frekuensi cutoff harus berupa angka.")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menerapkan filter: {e}")

    def plot_audio(self):
        self.ax1.clear()
        self.ax2.clear()

        if self.audio_data is not None:
            time_axis = np.linspace(0, len(self.audio_data) / self.sample_rate, num=len(self.audio_data))
            self.ax1.plot(time_axis, self.audio_data, label='Sinyal Asli')
            self.ax1.set_title('Sinyal Audio Asli (Domain Waktu)')
            self.ax1.set_xlabel('Waktu (s)')
            self.ax1.set_ylabel('Amplitudo')
            self.ax1.grid(True)
            
            fft_original = np.fft.fft(self.audio_data)
            freq_original = np.fft.fftfreq(len(self.audio_data), d=1/self.sample_rate)
            self.ax2.plot(freq_original[:len(freq_original)//2], np.abs(fft_original[:len(fft_original)//2]), label='Spektrum Asli')
            self.ax2.set_title('Spektrum Sinyal Audio')
            self.ax2.set_xlabel('Frekuensi (Hz)')
            self.ax2.set_ylabel('Magnitude')
            
            if self.filtered_data is not None:
                self.ax1.plot(time_axis, self.filtered_data, label='Sinyal Setelah Filter')
                
                fft_filtered = np.fft.fft(self.filtered_data)
                self.ax2.plot(freq_original[:len(freq_original)//2], np.abs(fft_filtered[:len(fft_filtered)//2]), label='Spektrum Setelah Filter')
            
            self.ax1.legend()
            self.ax2.legend()

        self.fig.tight_layout()
        self.canvas.draw()

    def update_cutoff_inputs(self, *args):
        """Menyesuaikan visibilitas input cutoff atas berdasarkan jenis filter yang dipilih."""
        if self.filter_type.get() == "band-pass":
            self.upper_cutoff_label.pack(side="left", padx=5, before=self.apply_button)
            self.upper_cutoff_entry.pack(side="left", padx=5, before=self.apply_button)
            self.cutoff_label.config(text="Frekuensi Cutoff Bawah (Hz):")
        else:
            self.upper_cutoff_label.pack_forget()
            self.upper_cutoff_entry.pack_forget()
            self.cutoff_label.config(text="Frekuensi Cutoff (Hz):")

def main():
    root = tk.Tk()
    app = AudioFilterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
