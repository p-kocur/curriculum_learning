# Zbierz argumenty wejściowe z konsoli
# Wywołaj eksperyment z podanymi argumentami
import argparse
import json

from src.run_training import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL experiment with specified scenario and teacher type.")
    parser.add_argument('--scenario', type=str, required=True, help='The scenario to run (e.g., bipedalwalker_easy, cartpole_hard).')
    parser.add_argument('--teacher_type', type=str, required=True, help='The type of teacher to use (e.g., alpgmm, oracle, random, rl).')
    
    args = parser.parse_args()

    if args.scenario == "bipedal_walker":
        with open("./env_config_bipedal_walker.json", 'r') as f:
            env_dict = json.load(f)
    else:
        raise ValueError("Unknown scenario: {}".format(args.scenario))
    
    with open("./rl_config.json", 'r') as f:
        rl_dict = json.load(f)

    with open("./curriculum_config.json", 'r') as f:
        curriculum_dict = json.load(f)
    
    curriculum_dict["teacher_type"] = args.teacher_type
    
    run(env_dict, rl_dict, curriculum_dict)



