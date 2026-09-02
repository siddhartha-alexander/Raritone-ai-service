from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "raritone_vton_trained.pth"
)

IMG_HEIGHT = 512
IMG_WIDTH = 384


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class RaritoneVTONNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = ConvBlock(7, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(
            256,
            512
        )

        self.up4 = nn.ConvTranspose2d(
            512,
            256,
            2,
            stride=2
        )
        self.dec4 = ConvBlock(
            512,
            256
        )

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            2,
            stride=2
        )
        self.dec3 = ConvBlock(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            2,
            stride=2
        )
        self.dec2 = ConvBlock(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            2,
            stride=2
        )
        self.dec1 = ConvBlock(
            64,
            32
        )

        self.output = nn.Conv2d(
            32,
            3,
            1
        )

    def forward(
        self,
        agnostic,
        cloth,
        cloth_mask
    ):

        masked_cloth = cloth * cloth_mask

        x = torch.cat(
            [
                agnostic,
                masked_cloth,
                cloth_mask
            ],
            dim=1
        )

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )

        d4 = self.up4(b)
        d4 = self.dec4(
            torch.cat(
                [d4, e4],
                dim=1
            )
        )

        d3 = self.up3(d4)
        d3 = self.dec3(
            torch.cat(
                [d3, e3],
                dim=1
            )
        )

        d2 = self.up2(d3)
        d2 = self.dec2(
            torch.cat(
                [d2, e2],
                dim=1
            )
        )

        d1 = self.up1(d2)
        d1 = self.dec1(
            torch.cat(
                [d1, e1],
                dim=1
            )
        )

        return torch.tanh(
            self.output(d1)
        )


image_transform = transforms.Compose([
    transforms.Resize(
        (IMG_HEIGHT, IMG_WIDTH)
    ),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.5,
            0.5,
            0.5
        ],
        std=[
            0.5,
            0.5,
            0.5
        ]
    )
])


mask_transform = transforms.Compose([
    transforms.Resize(
        (IMG_HEIGHT, IMG_WIDTH),
        interpolation=
        transforms.InterpolationMode.NEAREST
    ),

    transforms.ToTensor()
])


class VTONInference:

    def __init__(
        self,
        model_path=MODEL_PATH
    ):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Loading VTON model on "
            f"{self.device}"
        )

        self.model = (
            RaritoneVTONNet()
            .to(self.device)
        )

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model.eval()

        self.steps = checkpoint.get(
            "steps",
            "unknown"
        )

        self.final_loss = checkpoint.get(
            "final_loss",
            "unknown"
        )

        print(
            "VTON checkpoint loaded"
        )

        print(
            "Training steps:",
            self.steps
        )

        print(
            "Training loss:",
            self.final_loss
        )


    def predict(
        self,
        agnostic_path,
        garment_path,
        garment_mask_path,
        output_path
    ):

        agnostic = Image.open(
            agnostic_path
        ).convert("RGB")

        garment = Image.open(
            garment_path
        ).convert("RGB")

        garment_mask = Image.open(
            garment_mask_path
        ).convert("L")


        agnostic = (
            image_transform(
                agnostic
            )
            .unsqueeze(0)
            .to(self.device)
        )

        garment = (
            image_transform(
                garment
            )
            .unsqueeze(0)
            .to(self.device)
        )

        garment_mask = (
            mask_transform(
                garment_mask
            )
            .unsqueeze(0)
            .to(self.device)
        )


        with torch.no_grad():

            prediction = self.model(
                agnostic,
                garment,
                garment_mask
            )


        prediction = (
            prediction[0]
            .detach()
            .float()
            .cpu()
        )

        prediction = (
            prediction * 0.5
            + 0.5
        )

        prediction = (
            prediction
            .clamp(0, 1)
        )


        result = transforms.ToPILImage()(
            prediction
        )


        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        result.save(
            output_path
        )

        return {
            "success": True,
            "output": str(
                output_path
            ),
            "device": str(
                self.device
            )
        }


if __name__ == "__main__":

    inference = VTONInference()

    print(
        "Standalone VTON "
        "inference ready."
    )