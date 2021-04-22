from pytorch.models.abc_params.fully_connected.ipllr import FcIPLLR
import time
import click


@click.command()
@click.option('--n', '-N', required=False, type=click.INT, default=10,
              help='n')
@click.option('--word', '-W', required=False, type=click.STRING, default="Karl",
              help='word')
def main(n=10, word="Karl"):
    print('parameters of the run : n={}, word={}'.format(n, word))
    time.sleep(120)
    for i in range(n):
        print('Hello #{} {}'.format(i+1, word))
        time.sleep(4)
    print('\n ------ GoodBye ------')
    time.sleep(10)


if __name__ == '__main__':
    main()
