"""Utilities for parameterized testing"""
from parameterized import parameterized


def get_class_name(cls, num, params_dict):
    return "%s_%s" % (
        cls.__name__,
        parameterized.to_safe_name(params_dict["operator_cls"].__name__),
    )


def get_func_name(func, num, params_dict):
    return "%s_%s" %(
        func.__name__,
        parameterized.to_safe_name("_".join(str(x) for x in params_dict.args)),
    )


def generate_input_data(n_inputs, shape, dtype, xp):
    """Generates N random inputs with given shape and dtype"""
    inputs = []
    rng = xp.random.default_rng(seed=42)
    for i in range(n_inputs):
        curr_shape = get_item(shape, i)
        curr_dtype = get_item(dtype, i)
        real_values = rng.random(curr_shape, dtype="float64") * 1000
        if "complex" in curr_dtype:
            im_values = rng.random(curr_shape, dtype="float64") * 1000
            complex_values = real_values + 1j * im_values
            inputs.append(complex_values.astype(curr_dtype))
        else:
            inputs.append(real_values.astype(curr_dtype))
    return inputs


def get_item(item_container, index):
    """Access an item that may be on a list (item_container) or not"""
    if isinstance(item_container, list):
        return item_container[index]
    return item_container
