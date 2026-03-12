# Topic 3

## Task 1

```text
Sequential
TotalSeconds      : 318.409333
TotalMilliseconds : 318409.333

Parallel
TotalSeconds      : 257.349581
TotalMilliseconds : 257349.581
```

## Task 3

```text
Enter a math question ('quit' to exit):
> what is 157 times 2

Calling model...
Assistant: 157 times 2 is 314.

> quit
Goodbye!
```

## Task 4

```text
> What's the weather in San Francisco?
Assistant: The weather in San Francisco is sunny with a temperature of 72 F.

> What is the sin of the difference between the number of i's and the number of s's in Mississippi riverboats
Assistant: The number of 'i's is 5, the number of 's's is 5, the difference is 0, and sin(0) = 0.

> what is the count of characters of this text minus 5
Assistant: Please provide the text you'd like me to evaluate for the character count.

> this one
Assistant: The text "this one" contains 8 characters, including spaces.
```

## Task 5

```text
> What is (2*3) + 5?
Assistant: The expression evaluates to 11.
```

See `task5_portfolio.md` for the portfolio write-up.

## Task 6

The main missed parallelization opportunity is tool execution when the model requests multiple independent tools in the same turn. In `Task5.py`, `run_tools` executes each tool call serially, so independent calls could be executed concurrently before returning results to the model.
