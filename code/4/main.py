def heavy_function():
    lst = [i**2 for i in range(10**6)]  # memory + CPU work
    return sum(lst)
if __name__ == "__main__":
    result=heavy_function()
    print(result)