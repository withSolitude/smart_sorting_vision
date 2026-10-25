from vision.rules.box_manager import BoxManager


def main():

    manager = BoxManager()

    result = manager.classify(
        shape="三棱柱"
    )

    print(result)


if __name__ == "__main__":
    main()
