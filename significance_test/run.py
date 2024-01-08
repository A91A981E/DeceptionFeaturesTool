import liwc_test
import mfcc_test
import openface_csv_test
import pos_test

import json
import os


def run():
    result = {}
    output_path = os.path.join(os.getcwd(), "output", "sigtest")

    # liwc_res = liwc_test.run_test(output_path=output_path)
    # result.update(liwc_res)

    mfcc_res = mfcc_test.run_test(output_path=output_path)
    result.update(mfcc_res)

    openface_csv_res = openface_csv_test.run_test(output_path=output_path)
    result.update(openface_csv_res)

    # pos_res = pos_test.run_test(output_path=output_path)
    # result.update(pos_res)

    with open(os.path.join(output_path, "results.json"), "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    run()
