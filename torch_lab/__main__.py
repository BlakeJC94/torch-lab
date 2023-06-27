import fire

from . import train, build, __version__


def main():
    fire.Fire(
        dict(
            version=str(__version__),
            train=train,
            build=build,
        )
    )


if __name__ == "__main__":
    main()
