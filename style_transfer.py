import tensorflow as tf
import numpy as np
from PIL import Image
import time
import argparse
import cv2 


def load_and_preprocess_image(image_path):
  img = Image.open(image_path)

  #calculate new size, keep aspect ratio
  long = max(img.size) #img.size=(width, height)
  scale = max_dim / long
  new_size = (round(img.size[0] * scale), round(img.size[1]*scale))
  img = img.resize(new_size, Image.LANCZOS) # Use LANCZOS for high-quality resize

  #convert PIL image to a numpy array [height, width, color]
  img=tf.keras.preprocessing.image.img_to_array(img) 

  # Add a "batch dimension"
  # The model expects to see a *list* of images.
  # This turns our single image [height, width, 3]
  # into a "list" of one image: [1, height, width, 3]
  img = np.expand_dims(img, axis=0)

  # Pre-process the image for VGG19
  # This is a critical step! This function subtracts the
  # average color values (mean pixel) that the VGG model
  # was trained on and switches from RGB to BGR.
  # Our image is now in the *exact* format VGG expects.
  img = tf.keras.applications.vgg19.preprocess_input(img)

  return img

# These are the names of the VGG19 layers we'll use.
# These names are the standard, well-tested choices for this project.

# We'll use one deep layer for content
content_layer_name = 'block5_conv2'

#We'll use a set of layers from early to deep for style
style_layer_names = ['block1_conv1',
                     'block2_conv1',
                     'block3_conv1',
                     'block4_conv1',
                     'block5_conv1']

# A helper variable to store how many style layers we have
num_style_layers = len(style_layer_names)

def get_style_transfer_model(style_layers, content_layer):
    """
    Creates our "art critic" model. This model will take an image as
    input, and will output the internal activations from the
    style and content layers we've chosen.
    """
    # Load the VGG19 model *inside* the function to keep it self-contained
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    # Get the dictionary of all layer names and their
    # actual layer objects
    layer_outputs_by_name = dict([(layer.name, layer.output) for layer in vgg.layers])
    
    # Define the list of outputs we want to get
    style_outputs = [layer_outputs_by_name[name] for name in style_layers]
    content_output = layer_outputs_by_name[content_layer]
    
    # Combine all our desired outputs into one list
    model_outputs = style_outputs + [content_output]
    
    # Build the new Keras Model
    # Input: The standard VGG input
    # Output: The list of layer activations we just defined
    model = tf.keras.models.Model(inputs=vgg.input, outputs=model_outputs)
    
    return model

def gram_matrix(input_tensor):
  """
  Calculates the Gram Matrix (the "style signature") of a layer's output.
  The Gram Matrix measures the correlations between different filter responses.
  """

  # 1. Reshape the tensor
  # The input shape is (batch, height, width, channels).
  # We want to flatten height and width into one dimension.
  # New shape: (batch, num_locations, channels), where num_locations = height * width
  t = tf.reshape(input_tensor, [tf.shape(input_tensor)[0], -1, tf.shape(input_tensor)[3]])

  # 2. Transpose the tensor
  # We want to multiply channels by channels.
  # New shape: (batch, channels, num_locations)
  t_transpose = tf.transpose(t, [0, 2, 1])

  # 3. Calculate the Gram Matrix
  # This matrix multiplication (t_transpose @ t) creates a
  # (batch, channels, channels) tensor.
  # This is the correlation matrix.
  gram = tf.matmul(t_transpose, t)

  # 4. Normalize the Gram Matrix
  # We divide by the number of locations (h*w) to make sure
  # the loss isn't unfairly large just for large images.
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

  return gram / num_locations

def calculate_style_loss(style_target_features, generated_style_features):
  """
  Calculates the total style loss.
  It's the sum of the differences between the Gram Matrices
  of the style image and the generated image, for each layer.
  """

  total_loss = 0.0

  # We loop through each pair of layer features  
  # 'zip' pairs them up: (layer1_style, layer1_generated), (layer2_style, layer2_generated), ...
  for style_target, generated_output in zip(style_target_features, generated_style_features):

    #caculate the 'style signature' (gram matrix) for both
    gram_style = gram_matrix(style_target)
    gram_generated = gram_matrix(generated_output)

    #calculate the loss for thid *one* layer
    #this is the Mean Squared Error (the difference) between the two signatures
    layer_loss = tf.reduce_mean((gram_style - gram_generated)**2)

    # add this layer's loss to the total
    total_loss += layer_loss

  # avg the loss across all 5 layers
  total_loss /= len(style_target_features)

  return total_loss

def calculate_content_loss(content_target_feature, generated_content_feature):
    """
    Calculates the content loss.
    This is simply the Mean Squared Error between the feature maps
    of the content image and the generated image for our one content layer.
    """
    loss = tf.reduce_mean((content_target_feature - generated_content_feature)**2)
    return loss

def get_target_features(model, content_img, style_img):
    """
    Runs our content and style images through the 'art_critic_model'
    one time to get the "target" feature values we'll aim for.
    """
    # We combine them into one "batch"
    combined_input = tf.concat([content_img, style_img], axis=0)
    
    # The model returns 6 outputs (5 style, 1 content)
    outputs = model(combined_input)
    
    # Now, let's separate them
    style_outputs = outputs[:num_style_layers] 
    content_output = outputs[num_style_layers:]
    
    # We get the features for *only* the content image (index 0)
    # and *keep* the batch dimension (using 0:1).
    content_target = content_output[0][0:1] 
    
    # We get the features for *only* the style image (index 1)
    # and *keep* the batch dimension (using 1:2).
    style_targets = [style_layer[1:2] for style_layer in style_outputs]
    
    return content_target, style_targets

@tf.function
def train_step(image, tape, optimizer, content_target, style_targets, content_weight, style_weight):
    """
    Performs one step of optimization (one "painting" stroke).
    We pass all arguments so the function is self-contained.
    """
    # 1. Start the "tape recorder"
    with tape:
        
        # 2. Get the "reports" for our CURRENT image
        all_features = art_critic_model(image)
        style_features = all_features[:num_style_layers]
        content_features = all_features[num_style_layers:]
        
        # 3. Calculate the Loss
        current_content_loss = calculate_content_loss(content_target, content_features[0])
        current_style_loss = calculate_style_loss(style_targets, style_features)
        
        # 4. Calculate the Total Loss
        total_loss = (content_weight * current_content_loss) + (style_weight * current_style_loss)
                     
    # 5. Get the Gradients (The "direction" to paint)
    gradients = tape.gradient(total_loss, image)
    
    # 6. Apply the Gradients (Make the "stroke")
    optimizer.apply_gradients([(gradients, image)])
    
    # 7. Clip the image values to keep them valid
    # The VGG pre-processing expects values in a certain range.
    # This keeps our pixel values from "exploding"
    image.assign(tf.clip_by_value(image, -103.939, 151.061))
    
    return total_loss

def deprocess_image(processed_img):
    """
    This reverses the VGG pre-processing to turn our
    tensor back into a normal, viewable image.
    """
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0) # Remove the batch dimension (1, H, W, 3) -> (H, W, 3)
    
    # This is the *exact reverse* of tf.keras.applications.vgg19.preprocess_input
    
    # 1. Add the "mean pixel" values back (which were subtracted)
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    
    # 2. Flip from BGR back to RGB
    x = x[:, :, ::-1]
    
    # 3. Clip values to be valid image data (0 to 255)
    x = np.clip(x, 0, 255).astype('uint8')
    
    return x




# This code only runs when you execute the script directly
if __name__ == '__main__':

    # Set up all our command-line arguments (our "knobs")
    # This uses the 'argparse' library we imported
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('--content_path', type=str, required=True, help='Path to the content image.')
    parser.add_argument('--style_path', type=str, required=True, help='Path to the style image.')
    parser.add_argument('--output_path', type=str, default='my_masterpiece.jpg', help='Path to save the output image.')
    
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to run.')
    parser.add_argument('--steps_per_epoch', type=int, default=100, help='Number of steps per epoch.')
    
    # These are the "knobs" we tuned in Colab!
    parser.add_argument('--style_weight', type=float, default=1.0, help='Weight for the style loss. Try 1.0, 5.0, 10.0')
    parser.add_argument('--content_weight', type=float, default=1e4, help='Weight for the content loss. Keep at 1e4.')
    parser.add_argument('--max_dim', type=int, default=512, help='Resize images to this max dimension. 1024 for high-res.')
    
    # This line reads all the arguments from the command line
    args = parser.parse_args()

    print("--- 🎨 Neural Style Transfer ---")

    # 1. Load images (using args.content_path)
    print(f"Loading content image: {args.content_path}")
    content_image = load_and_preprocess_image(args.content_path, args.max_dim)
    
    print(f"Loading style image: {args.style_path}")
    style_image_orig = load_and_preprocess_image(args.style_path, args.max_dim)

    # 2. Resize style image (so that style becomes same size of content img so that they both can add up)
    target_shape = content_image.shape[1:3]
    style_image_resized = tf.image.resize(style_image_orig, target_shape)

    # 3. Build model 
    art_critic_model = get_style_transfer_model(style_layer_names, content_layer_name)

    # 4. Get target features 
    print("Extracting target features...")
    content_target_features, style_target_features = get_target_features(
        art_critic_model, 
        content_image, 
        style_image_resized
    )

    # 5. Set up for training 
    image_to_be_generated = tf.Variable(content_image, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    # We create the GradientTape *outside* the function for optimization
    tape = tf.GradientTape()

    print(f"🎨 Starting painting... Total steps: {args.epochs * args.steps_per_epoch}")
    start_time = time.time()

    # 6. The main loop 
    for epoch in range(args.epochs):
        print(f"\n--- Starting Epoch {epoch + 1}/{args.epochs} ---")
        for step in range(args.steps_per_epoch):
            total_loss = train_step(
                image_to_be_generated, 
                tape, 
                optimizer, 
                content_target_features, 
                style_target_features,
                args.content_weight,  
                args.style_weight   
            )

            if (step + 1) % 50 == 0:
                print(f"  Step {step + 1}/{args.steps_per_epoch}, Total Loss: {total_loss:.2f}")

                # 1. Deprocess the image to a viewable format (0-255, RGB)
                img_array_rgb = deprocess_image(image_to_be_generated.numpy())
                # Convert from RGB (what deprocess gives) to BGR (what OpenCV needs)
                img_array_bgr = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2BGR)

                # 2. Get the dimensions of our painting
                h, w, c = img_array_bgr.shape

                # 3. Create a new, blank "header" canvas
                header_height = 60  # 60 pixels tall
                header = np.zeros((header_height, w, 3), dtype="uint8") # Black background

                # 4. Define our text, font, and color
                text_step = f"Step: {step + 1}/{args.steps_per_epoch}"
                text_loss = f"Loss: {total_loss:.2f}"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                text_color = (255, 255, 255) # White

                # 5. Get the text size to center it
                text_step_size, _ = cv2.getTextSize(text_step, font, font_scale, font_thickness)
                text_loss_size, _ = cv2.getTextSize(text_loss, font, font_scale, font_thickness)

                # Calculate X coordinate to be in the center
                text_step_x = (w - text_step_size[0]) // 2
                text_loss_x = (w - text_loss_size[0]) // 2

                # 6. Put the centered text onto our black "header"
                cv2.putText(header, text_step, (text_step_x, 25), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                cv2.putText(header, text_loss, (text_loss_x, 50), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

                # 7. Use NumPy to stack the header on top of the image
                combined_image = np.vstack((header, img_array_bgr))

                # 8. Display the combined image in a window named 'Painting...'
                cv2.imshow('Painting in Progress', combined_image)

                # 4. Wait 1ms. 
                cv2.waitKey(1)

    end_time = time.time()
    print(f"\n✅ 'Painting' complete in {end_time - start_time:.1f} seconds.")

    # 7. Save the final image 
    final_image_array = deprocess_image(image_to_be_generated.numpy())
    final_image = Image.fromarray(final_image_array)
    final_image.save(args.output_path) # Save to the path from our 'knob'

    cv2.destroyAllWindows()

    print(f"Image saved as '{args.output_origin_path}'")

