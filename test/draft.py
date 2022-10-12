from torchvision import models


def main():
    model = models.resnet50(pretrained=True)

    for k, v in model.items():
        print(f"k = {k}")

    print(f"kkkkkkkkkkkkkkkkkkkkk")


if __name__ == '__main__':
    main()