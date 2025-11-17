# 🎨 Neural Style Transfer (Command-Line Tool)

This project is a command-line tool that implements **Neural Style Transfer** in Python using TensorFlow and OpenCV.

It takes a **content image** (like a photo) and a **style image** (like a painting) and generates a new image by "painting" the content with the style of the other.

This implementation is based on the original paper: [A Neural Algorithm of Artistic Style by Gatys et al.](https://arxiv.org/abs/1508.06576)



---

## 🚀 Features

* **Real-time Preview:** Uses an OpenCV window to show the image being "painted" live, step-by-step.
* **Live Stats:** Draws the current step and loss directly onto the preview window, not in the terminal.
* **CLI Control:** All "knobs" are controlled via command-line arguments:
    * Adjust style vs. content strength (`--style_weight`)
    * Change image resolution (`--max_dim`)
    * Set training time (`--epochs`)

---

## 🖼️ Example Result

Here is an example of a photo of a building "painted" in the style of Van Gogh's "Starry Night".
<img src="https://github.com/user-attachments/assets/1d988c39-ed7b-4507-b6f4-329570f7b736" alt="Masterpiece" width="400px">


> **(IMPORTANT: You must add your own image here!)**
>
> 1. Run your script to create a cool `my_masterpiece.jpg`.
> 2.  Upload that JPG to your GitHub repository.
> 3.  Edit this README file on GitHub and drag-and-drop your image right here to replace this text.

---

## 🛠️ How to Use

### 1. Prerequisites

* Python 3.7+
* An environment with a (preferably NVIDIA) GPU is highly recommended, as this is a computationally heavy task.

### 2. Installation

First, clone this repository to your local machine:

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
```

(You will need to replace YOUR_USERNAME and YOUR_REPOSITORY_NAME with your actual GitHub details after you upload it.)

### 3. Setup Virtual Environment (Recommended)

It's a best practice to create and activate a virtual environment so you don't install packages globally.

```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install Dependencies

Install all the required libraries from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 5. Run the Script!

You are now ready to run the script. The two required arguments are `--content_path` and `--style_path`.

A new window will pop up showing you the painting process. The final image will be saved as `my_masterpiece.jpg` (or whatever you name it).

**Basic Example:**

```bash
python style_transfer.py --content_path "path/to/my_photo.jpg" --style_path "path/to/my_painting.jpg"
```

**Advanced Example (More "Stylish" and Higher-Res):**

This command increases the style's influence (--style_weight) and runs for longer (--epochs), creating a 1024px image (--max_dim).

```
python style_transfer.py --content_path "my_photo.jpg" --style_path "my_painting.jpg" --output_path "high_res_art.jpg" --max_dim 1024 --style_weight 5.0 --epochs 15
```

---

## ⚙️ Command-Line Arguments

Here are all the "knobs" you can tune, explained in a table:

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--content_path` | str | **Required** | Path to the content image (e.g., "photo.jpg"). |
| `--style_path` | str | **Required** | Path to the style image (e.g., "art.jpg"). |
| `--output_path` | str | "my_masterpiece.jpg"| Name of the final file to save. |
| `--epochs` | int | 10 | How many "passes" to run the painter for. |
| `--steps_per_epoch` | int | 100 | How many "strokes" to paint per pass. |
| `--style_weight` | float | 1.0 | How strong the style is. **Try `5.0` or `10.0` for a *much* stronger effect.** |
| `--content_weight`| float | 10000 | How strongly to preserve the content. (Best to leave this at `1e4`). |
| `--max_dim` | int | 512 | Resizes the longest side of the image to this. `1024` gives high-res results but is ~4x slower. |
