"""Accept movement keys from both AZERTY and QWERTY layouts."""

KEY_COMMANDS = {ord("W"): "STRAIGHT", ord("Z"): "STRAIGHT",
                ord("A"): "LEFT", ord("Q"): "LEFT",
                ord("D"): "RIGHT", ord("S"): "STOP", 32: "STOP"}


def command_for_key(key):
    return KEY_COMMANDS.get(key)
