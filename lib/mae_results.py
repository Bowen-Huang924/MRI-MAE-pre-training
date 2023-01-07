import logging

import lib.utils
from lib import mae

if __name__ == '__main__':
    # 设置日志消息格式
    logging.basicConfig(format="%(asctime)s-%(message)s", level=logging.INFO)

    configs = [
        # '../configs/exp/transresunet_v2-ori.yml',
        '../configs/exp/mae_result.yml',
    ]

    # 读取运行参数
    for config in configs:
        # 加载配置文件
        cfg = lib.utils.load_yaml_config(config)
        k_fold = 1
        exp_name = cfg["exp_name"]
        print("k_fold start")
        for i in range(k_fold):
            # 实例化训练流程并开始训练

            cfg["fold_id"] = i+1
            cfg["exp_name"] = exp_name + "_fold{}".format(i+1)
            pipeline = mae.PipeLine(cfg)
            pipeline.run()
        print("k_fold end")
