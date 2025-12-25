import numpy as np
import soundfile as sf
import os

# הגדרות שמע
SAMPLE_RATE = 16000 # סטנדרט של מודלים לדיבור
DURATION = 5        # שניות
OUTPUT_DIR = "ml-core/data-pipelines/synthetic_data"

def create_synthetic_audio(filename):
    # 1. יצירת ציר הזמן
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    
    # 2. יצירת ה"סיגנל" (דמוי דיבור - תדר משתנה)
    # נחבר כמה תדרים כדי שזה לא יהיה צפצוף משעמם
    signal = 0.5 * np.sin(2 * np.pi * 440 * t) + \
             0.3 * np.sin(2 * np.pi * 880 * t)
    
    # 3. יצירת "רעש" (רעש לבן אקראי)
    noise = np.random.normal(0, 0.2, signal.shape)
    
    # 4. הערבוב (אודיו מלוכלך)
    noisy_signal = signal + noise
    
    # נרמול (שלא יחרוג ממינוס 1 עד 1)
    max_val = np.max(np.abs(noisy_signal))
    if max_val > 0:
        noisy_signal = noisy_signal / max_val
        signal = signal / np.max(np.abs(signal)) # נרמול גם לנקי

    return noisy_signal, signal

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Generating synthetic audio in {OUTPUT_DIR}...")
    
    # ניצור 10 דוגמאות
    for i in range(10):
        noisy, clean = create_synthetic_audio(f"sample_{i}")
        
        # שמירה לקבצים
        sf.write(f"{OUTPUT_DIR}/noisy_{i}.wav", noisy, SAMPLE_RATE)
        sf.write(f"{OUTPUT_DIR}/clean_{i}.wav", clean, SAMPLE_RATE)
        
    print("Done! Data is ready for training.")

if __name__ == "__main__":
    main()
