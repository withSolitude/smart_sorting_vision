"""
Controller interface test.

Uses mock modules.
"""


from vision.system.sorting_controller import SortingController


class MockSTM32:

    def send_start(self):
        print("START")


    def send_stop(self):
        print("STOP")


    def send_pick_target(self,x,y):
        print("PICK",x,y)


    def send_sort_result(self,box):
        print("BOX",box)



def main():

    print("SortingController module loaded")


if __name__=="__main__":
    main()
