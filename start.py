import sys


if __name__ == "__main__":
    mode = sys.argv[1] if len(
        sys.argv) > 1 and sys.argv[1].strip() else "itf"

    if mode == "bdh":
        from factual_models.bdh_train import main as bdh_main
        bdh_main()
    elif mode == "tf":
        from factual_models.tf_train import main as tf_main
        tf_main()
    elif mode == "ibdh":
        from factual_models.bdh_infer import main as infer_main
        infer_main()
    elif mode == "itf":
        from factual_models.tf_infer import main as infer_main
        infer_main()
    elif mode == "vbdh":
        from factual_models.viz_bdh_infer import main as infer_main
        infer_main()
    elif mode == "cbdh":
        from counting_models.count_bdh_train import main as count_main
        count_main()

    elif mode == "icbdh":
        from counting_models.count_bdh_infer import main as count_main
        count_main()
    elif mode == "ctf":
        from counting_models.count_tf_train import main as count_main
        count_main()

    elif mode == "ictf":
        from counting_models.count_tf_infer import main as count_main
        count_main()
    else:
        raise ValueError(f"Unknown mode: {mode}. Use bdh, tf, or inference.")
