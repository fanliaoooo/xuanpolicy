'''
Repo: XuanCe

Example for single agent DRL:
$ python main.py --method dqn --env-name toy_env/CartPole-v0

'''
import argparse
from xuance import get_runner


def parse_args():
    parser = argparse.ArgumentParser("Run a demo.")
    parser.add_argument("--method", type=str, default="a2c")
    parser.add_argument("--env", type=str, default="classic_control")
    parser.add_argument("--env-id", type=str, default="Pendulum-v1")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


if __name__ == '__main__':
    parser = parse_args()
    runner = get_runner(method=parser.method,
                        env=parser.env,
                        env_id=parser.env_id,
                        parser_args=parser,
                        is_test=parser.test)
    runner.run()
