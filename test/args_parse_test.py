import argparse
parser = argparse.ArgumentParser(prog='MRI_Trans_GAN',allow_abbrev=False,fromfile_prefix_chars='@')
args = parser.add_argument('action',choices=['initial','train','test','debug',"train-and-test"],type=str.lower)
args = parser.add_argument("--workspace",type=str),
args = parser.add_argument("--init",type=str.lower)
args = parser.add_argument("--indicator",type=str.lower)
args = parser.add_argument("--global_random_seed",type=int,help="全局随机种子 其余部分")

print(args.default)
