
from vision.storage.object_info import ObjectInfo
from vision.storage.storage_manager import StorageManager



def main():

    manager = StorageManager(6)


    test_objects = [

        ObjectInfo(
            color="gray",
            shape="三棱柱",
            qr_present=True
        ),

        ObjectInfo(
            color="gray",
            shape="三棱柱",
            qr_present=True
        ),

        ObjectInfo(
            color="blue",
            shape="正方体",
            qr_present=True
        ),

    ]


    for obj in test_objects:

        result = manager.allocate(obj)

        print(
            obj,
            "=>",
            result
        )


    print("\nStorage status:")

    for item in manager.get_status():

        print(item)



if __name__ == "__main__":
    main()
