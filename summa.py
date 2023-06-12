import sys

def create_text_box(prompt):
    if sys.platform.startswith('win'):
        import msvcrt

        sys.stdout.write(prompt)
        sys.stdout.flush()
        input_str = ""
        while True:
            char = msvcrt.getwch()
            if char in '\r\n':
                break
            elif char == '\b':
                if input_str:
                    sys.stdout.write('\b \b')
                    sys.stdout.flush()
                    input_str = input_str[:-1]
            else:
                sys.stdout.write(char)
                sys.stdout.flush()
                input_str += char

        sys.stdout.write('\n')
        return input_str
    else:
        return input(input_str)

# Example usage:
name = create_text_box("Enter your name: ")
print("Hello, " + name + "!")
