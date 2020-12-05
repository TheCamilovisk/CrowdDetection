import streamlit as st
import numpy as np
import cv2
from model import CSRNet
from torchvision import transforms
import torch
import PIL.Image as Image
from os import path
import matplotlib.pyplot as plt
from matplotlib import cm as c

transform = None
model = None


def run_crowd_counter(img, transform, model):
    img = transform(img).cuda()
    output = model(img.unsqueeze(0))
    crowd_count = int(output.detach().cpu().sum().numpy())
    prob_map = np.asarray(
        output.detach()
        .cpu()
        .reshape(output.detach().cpu().shape[2], output.detach().cpu().shape[3])
    )
    return prob_map, crowd_count


def main():
    # Render the readme as markdown
    readme_text = st.markdown(get_file_content_as_string("README.md"))

    transform, model = initialize_model()

    # Add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox(
        "Choose app mode", ["Show instructions", "Run the app"]
    )
    if app_mode == "Show instructions":
        st.sidebar.success('To continue slect "Run the app".')
    elif app_mode == "Run the app":
        readme_text.empty()
        run_the_app(transform, model)


@st.cache(show_spinner=False)
def initialize_model():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model = CSRNet().cuda()
    checkpoint = torch.load(path.join("checkpoints", "model_best.pth.tar"))
    model.load_state_dict(checkpoint["state_dict"])

    return transform, model


def run_the_app(transform, model):
    st.markdown("# Run the app")

    # Draw the UI elements to process the crowd counter.
    img, img_path = image_selector()
    if img is not None:
        process_ui(img, img_path, transform, model)


def process_ui(img, img_path, transform, model):
    h, w = img.shape[:2]

    st.subheader("Input image")
    st.write(img_path)

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img)
    st.pyplot(fig)

    process_buttom = st.sidebar.button("RUN")
    if process_buttom:
        prob_map, crowd_count = run_crowd_counter(img, transform, model)
        st.markdown(f"Estimated people count: **{crowd_count}**")

        st.subheader("Probability map")
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        prob_map = cv2.resize(prob_map, (w, h))
        ax.imshow(prob_map)
        st.pyplot(fig)


def image_selector():
    # Ad-hoc file selector.
    img_path = st.sidebar.text_input("Enter a image path:")
    img = cv2.imread(img_path)

    if img is None:
        return None, None

    h, w = img.shape[:2]
    if w > 800:
        img = cv2.resize(img, (0, 0), fx=(800 / w), fy=(800 / w))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_path


@st.cache(show_spinner=False)
def get_file_content_as_string(readme_path):
    with open("README.md") as file:
        file_contents = file.read()
    return file_contents


if __name__ == "__main__":
    main()
