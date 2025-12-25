import os
import sys
import jax
import jax.numpy as jnp
import tensorflow as tf
from jax.experimental import jax2tf
from flax import serialization
import numpy as np

# 1. הוספת הנתיב למודל המקורי
sys.path.append(os.path.join(os.path.dirname(__file__), '../training'))
from model import DenoisingAutoencoder

# הגדרות
WEIGHTS_PATH = '../training/denoiser_weights.msgpack'
OUTPUT_PATH = 'denoiser.tflite'
INPUT_SHAPE = (1, 16000)  # שנייה אחת של אודיו

def main():
    print("--- Starting JAX to TFLite Conversion ---")

    # 2. טעינת המודל והמשקולות (כמו במחברת)
    print("1. Loading JAX model...")
    model = DenoisingAutoencoder()
    
    # יצירת משתנה דמי לאתחול
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones(INPUT_SHAPE)
    params = model.init(rng, dummy_input)['params']

    # קריאת המשקולות מהקובץ
    with open(os.path.join(os.path.dirname(__file__), WEIGHTS_PATH), 'rb') as f:
        params = serialization.from_bytes(params, f.read())

    # 3. הגדרת פונקציית ההרצה (Inference)
    # אנו יוצרים פונקציה שמקבלת קלט ומחזירה פלט, כשהמשקולות "מוקפאות" בפנים
    def predict_fn(input_data):
        return model.apply({'params': params}, input_data)

    # 4. המרה ל-TensorFlow Function
    # jax2tf הופך את הקוד של JAX לגרף של TensorFlow
    print("2. Converting to TensorFlow graph...")
    tf_predict = jax2tf.convert(predict_fn, with_gradient=False)

    # 5. המרה ל-TFLite
    print("3. Converting to TFLite flatbuffer...")
    
    # אנו יוצרים "פונקציה קונקרטית" שמקובעת לגודל הקלט שלנו
    concrete_func = tf.function(tf_predict).get_concrete_function(
        tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.float32))

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    
    # אופטימיזציה (לא חובה, אבל מומלץ למובייל)
    # זה מקטין את הקובץ ומאיץ ריצה, לפעמים במחיר דיוק זעיר
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()

    # 6. שמירה לקובץ
    output_file = os.path.join(os.path.dirname(__file__), OUTPUT_PATH)
    with open(output_file, 'wb') as f:
        f.write(tflite_model)

    print(f"SUCCESS! Model saved to: {output_file}")
    print(f"File size: {len(tflite_model) / 1024:.2f} KB")

if __name__ == '__main__':
    main()
