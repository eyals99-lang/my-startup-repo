import jax
import jax.numpy as jnp
from flax import linen as nn

class DenoisingAutoencoder(nn.Module):
    """A simple MLP autoencoder for audio denoising."""
    
    @nn.compact
    def __call__(self, x):
        # x shape: [batch_size, input_length]
        
        # Encoder: דחיסה של המידע
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        
        # Bottleneck: המידע הכי מזוקק
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        
        # Decoder: ניסיון לשחזר את האות המקורי (בלי הרעש)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)
        
        # שכבת היציאה: חזרה לגודל המקורי
        # אנחנו לא שמים Activation כאן כדי לאפשר ערכים שליליים (כמו באודיו)
        x = nn.Dense(features=16000)(x) # נניח שאנחנו מעבדים שניה אחת כל פעם
        
        return x

# פונקציית עזר לבדיקה שהמודל מתקמפל
def print_model_summary():
    model = DenoisingAutoencoder()
    # יצירת דאטה דמי לבדיקת אתחול
    key = jax.random.PRNGKey(0)
    dummy_input = jax.random.normal(key, (1, 16000)) # Batch size 1, 1 sec audio
    
    print("Initializing model...")
    params = model.init(key, dummy_input)
    print("Model initialized successfully!")
    
    # הרצה ניסיונית
    output = model.apply(params, dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    print_model_summary()
