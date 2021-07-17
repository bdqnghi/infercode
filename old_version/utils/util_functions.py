def form_tbcnn_model_path(opt):
    model_traits = {}
    model_traits["parser"] = str(opt.parser)
    model_traits["type"] = str(opt.node_type_dim)
    model_traits["token"] = str(opt.node_token_dim)
    model_traits["conv_output"] = str(opt.conv_output_dim)
    model_traits["node_init"] = str(opt.node_init)
    model_traits["num_conv"] = str(opt.num_conv)
    

    model_path = []
    for k, v in model_traits.items():
        model_path.append(k + "_" + v)
    
    return "tbcnn" + "_" + "-".join(model_path)
