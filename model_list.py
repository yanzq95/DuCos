
def import_model(args):
    print("=> creating model {}".format(args.model_name))

    if args.model_name == 'DuCos':
        from model.ours.ducos import depthprompting
        args.prop_kernel = 9
        args.prop_time = 18
        args.conf_prop = True
        args.loss = 'L1L2'
        model = depthprompting(args)
    else:
        print("Check Model Name ! !")
        raise NotImplementedError
    return model
    