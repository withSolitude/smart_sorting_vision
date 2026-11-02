from runtime.sorting_runtime import SortingRuntime


def main():
    runtime = SortingRuntime(
        None,
        None,
        None,
        None,
        None,
        None
    )

    runtime.start()
    print('SortingRuntime structure OK')
    print(runtime.state)


if __name__ == '__main__':
    main()
