from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf

def get_init_fn_for_scaffold(model_dir,
                             checkpoint_path,
                             model_scope,
                             checkpoint_model_scope,
                             checkpoint_exclude_scopes,
                             ignore_missing_vars,
                             name_remap=None):

    if tf.train.latest_checkpoint(model_dir):
        tf.logging.info('Ignoring --checkpoint_path because a checkpoint'
                        ' already exists in %s.' % model_dir)
        return None

    # TL;DR: exclusion_scopes = [] version of checkpoint_exclude_scopes
    exclusion_scopes=[]
    if checkpoint_exclude_scopes:
        exclusion_scopes=\
            [scope.strip() for scope in checkpoint_exclude_scopes.split(',')]

    # 1: exclude all graph trainable variables (network weights, ...) that
    # reside in excluded scopes
    # E.g., we would like to exclude all prediction layers, ... from
    # TextBoxes++'s graph trainable variables when we finetune it from VGG16's
    # model file.
    variables_to_restore=[]
    for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        excluded = False
        for exclusion in exclusion_scopes:
            if exclusion in var.op.name:#.startswith(exclusion):
                excluded=True
                break
        if not excluded:
            variables_to_restore.append(var)

    # count total trainable parameters
    print('-----------------------------------------------------------------')
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parameters = 1
        for dim in shape:
            print(dim)
            variable_parameters *= dim.value
        print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)

    # 2: replace model scope name in graph with model scope name in model file
    # E.g., scope_name: textboxes_384, checkpoint_model_scope: vgg_16
    if checkpoint_model_scope is not None:
        if checkpoint_model_scope.strip() == '':
            variables_to_restore=\
                {var.op.name.replace(model_scope + '/', ''): var\
                 for var in variables_to_restore}
        else:
            variables_to_restore=\
                {var.op.name.replace(model_scope,
                                     checkpoint_model_scope.strip()): var\
                 for var in variables_to_restore}

        # 3: TL;DR ==> currently not used
        if name_remap is not None:
            renamed_variables_to_restore=dict()
            for var_name, var in variables_to_restore.items():
                found=False
                # k: old_name of graph's variables (name in graph we build)
                # v: new_name of graph's variables (name in model file)
                for k, v in name_remap.items():
                    if k in var_name:
                        renamed_variables_to_restore[var_name.replace(k, v)]=\
                            var
                        found=True
                        break
                if not found:
                    renamed_variables_to_restore[var_name]=var
            variables_to_restore=renamed_variables_to_restore

    # TL;DR
    checkpoint_path=\
        tf.train.latest_checkpoint(checkpoint_path)\
        if tf.gfile.IsDirectory(checkpoint_path)\
        else checkpoint_path # full path is provided

    tf.logging.info('Fine-tuning from {}. Ignoring missing vars: {}.'.format(
        checkpoint_path,
        ignore_missing_vars))

    if not variables_to_restore:
        raise ValueError('variables_to_restore cannot be empty')

    # 4
    # compare vars from model file (what we have) with vars from graph (what
    # we expect to have) to know which var is missing and which var is
    # actually restored to train.
    if ignore_missing_vars:
        reader=tf.train.NewCheckpointReader(checkpoint_path)
        if isinstance(variables_to_restore, dict):
            var_dict=variables_to_restore
        else:
            var_dict={var.op.name: var for var in variables_to_restore}
        available_vars={}
        for var in var_dict:
            if reader.has_tensor(var):
                available_vars[var]=var_dict[var]
            else:
                tf.logging.warning('Variable %s missing in checkpoint %s.',
                                   var,
                                   checkpoint_path)
        variables_to_restore=available_vars
    if variables_to_restore:
        saver=tf.train.Saver(variables_to_restore, reshape=False)
        saver.build() # create save/restore graph nodes.
        def callback(scaffold, session):
            saver.restore(session, checkpoint_path)
        return callback
    else:
        tf.logging.warning('No Variables to restore.')
        return None
