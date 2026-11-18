"""Simple script to generate linedividers that are perfectly centered."""

def main():
    max_size = len("--------------------------------------------------------------------------------------")
    # This was tested, for the old mfc gui, this is the maximum length displayed in the keyvalue table

    print(f"Message maximum length: {max_size}")
    
    title = input("Please input the category name: ")

    fill_size = max_size - len(title)
    if fill_size < 0:
        print("Category name too long!")
        return

    l_size = fill_size // 2
    r_size = fill_size // 2 + fill_size % 2
    print(f"L-size: {l_size} | R-size: {r_size}")

    print("-" * l_size + title + ("-" * r_size))


if __name__ == "__main__":
    main()