import fire

from . import train, test, build, __version__


def main():
    fire.Fire(
        dict(
            version=str(__version__),
            train=train,
            test=test,
            build=build,
        )
    )


if __name__ == "__main__":
    main()
