import keyboard

def main():
    while(True):
        if (keyboard.is_pressed('ctrl+q')):
            print("hit")
        else:
            print("not hit")
if __name__ == "__main__":
    main()

