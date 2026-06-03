# WAL (Waveform Analysis Language) — Reference Guide v0.8.2

> Source: https://wal-lang.org/documentation/  
> WAL is a Lisp-style domain-specific language (DSL) for querying, analyzing, and debugging hardware simulation waveforms (VCD, FST files). All expressions use **prefix S-expression syntax**: `(operator arg1 arg2 ...)`.

---

## Quick-Start Pattern

```wal
; Load a waveform, inspect signals, find a condition
(load "trace.vcd")        ; loads as t0 by default
SIGNALS                   ; lists all signal names
INDEX                     ; current time index (starts at 0)
tb.clk                    ; read signal value at current INDEX
(find (= tb.state 3))     ; returns list of all indices where condition is true
```

---

## 1. Syntax Basics

WAL uses **S-expressions** (like Lisp/Scheme). All operators and function calls are in **prefix notation**:

```wal
(+ 1 2 3)          ; → 6
(= a b)            ; equality check
(!= a b)           ; inequality
(! expr)           ; logical NOT
(&& a b c)         ; logical AND (supports multiple args)
(|| a b c)         ; logical OR
```

**Literals:**
- Integers: `1`, `0xff`
- Strings: `"text"`
- Booleans: `#t` (true), `#f` (false) — also `1`/`0`
- Symbols: bare words like `clk`, `tb.reset`

**Signal names** that contain dots (hierarchy separators) are written bare: `top.module.signal`

---

## 2. Waveform Loading & Navigation

### load / unload

```wal
(load "file.vcd")          ; load waveform, auto-assigned id (t0, t1, ...)
(load "file.vcd" myTrace)  ; load with explicit id
(unload myTrace)           ; remove from kernel
```

### INDEX and stepping

`INDEX` is a **special variable** — the current time pointer into the waveform.

```wal
INDEX                      ; read current index
(step)                     ; advance all traces by 1
(step 5)                   ; advance all traces by 5
(step myTrace 3)           ; advance specific trace by 3
                           ; returns #f if end of trace reached
```

### timeframe — local time operations

Saves the current INDEX, runs body, then restores INDEX. Use for lookahead without losing position.

```wal
(timeframe
  (while (! ready) (step))
  (print INDEX))
; INDEX is restored after timeframe exits
```

---

## 3. Accessing Signals

Signals behave like variables whose value depends on the current `INDEX`.

```wal
tb.clk             ; read signal tb.clk at current INDEX
(get "tb.clk")     ; same, using string name
```

### reval / @ — read at offset

```wal
(reval tb.clk -1)  ; read tb.clk at INDEX-1
tb.clk@-1          ; shorthand for the above
tb.clk@1           ; read at INDEX+1
INDEX@-1           ; → current INDEX - 1
INDEX@(+ 2 2)      ; offset can be an expression
```

### slice — bit extraction

```wal
(slice signal upper lower)   ; extract bits [upper:lower] from signal value
```

### SIGNALS — list all signals

```wal
SIGNALS            ; returns list of all signal names in loaded waveform
```

---

## 4. Searching and Querying

### find — get all matching indices

```wal
(find cond)        ; returns list of all INDEX values where cond is true

; Examples:
(find (= clk 1))                    ; all indices where clk is high
(find (&& comp1.ack comp1.req))     ; all indices where both signals high
(find (> data 100))                 ; all indices where data > 100
```

### count — count matching indices

```wal
(count cond)       ; returns number of indices where cond is true
(count (= clk 1))
```

### whenever — execute body at matching indices

Walks the entire waveform and evaluates `body` at every index where `cond` is true.

```wal
(whenever cond body+)

; Example: print INDEX and data whenever a valid transaction occurs
(whenever (&& clk (! reset) ready valid)
  (print INDEX ":" data))
```

---

## 5. Program State

### let — local bindings

```wal
(let ([x 10] [y 12])
  (+ x y))          ; → 22

(let ([x 10] [y x]) ; later bindings can use earlier ones
  (+ x y))          ; → 20
```

### define — global binding

```wal
(define x 10)
(define y (+ x x))
```

### set! — update existing binding

```wal
(define x 10)
(set! x (+ x x))   ; x is now 20
```

---

## 6. Control Flow

### if

```wal
(if cond then else)
; only single expressions for then/else — wrap multiple in (do ...)
(if a[2]
  (do (print "yes") (set! a (list)))
  (do (print "no")  (set! a (list))))
```

### when / unless

```wal
(when cond body+)    ; evaluate body if cond is truthy
(unless cond body+)  ; evaluate body if cond is falsy
```

### cond — multi-branch

```wal
(cond
  [(= n 1) 1]
  [(= n 2) 1]
  [#t (+ (fib (- n 1)) (fib (- n 2)))])  ; #t = default case
```

### case — value switch

```wal
(case (+ a b)
  [1 "one"]
  [2 "two"]
  [default "> two"])
```

### do — sequence block

```wal
(do expr1 expr2 expr3)  ; evaluates all, returns last result
```

---

## 7. Functions

### defun — named function

```wal
(defun times-two [n] (* n 2))
(times-two 5)   ; → 10

; Variadic (all args become a list):
(defun sum-all xs (fold + 0 xs))
(sum-all 1 2 3 4)
```

### fn — anonymous function (lambda)

```wal
((fn [a b] (+ a b)) 1 2)    ; → 3

; Closures work:
(defun make-counter [name]
  (define cnt 0)
  (fn [] (set! cnt (+ cnt 1))
         (print name ": " cnt)))
```

---

## 8. Groups and Scopes

Groups let you write **generic analysis** over structurally similar signals (e.g., multiple AXI handshake interfaces).

### groups — auto-discover signal prefixes

```wal
(groups "valid" "ready")
; returns all prefixes P such that P+"valid" and P+"ready" are valid signals
; e.g., → ("top.in_" "top.out_")
```

### in-groups — iterate over groups

When executing in a group, `#signal` is expanded to `currentGroup + signal`.  
`CG` is a special variable that holds the **current group** name.

```wal
(in-groups '("top.in_" "top.out_")
  (print CG ":")
  (whenever (&& top.clk (! top.reset) #ready #valid)
    (print INDEX ":" #data)))
; #ready → top.in_ready inside group "top.in_"
; #ready → top.out_ready inside group "top.out_"
```

### resolve-group / # macro

```wal
(resolve-group #valid)   ; evaluates CG + "valid" as a signal
#valid                   ; shorthand — same as (resolve-group #valid)
```

### in-scope / in-scopes / all-scopes

```wal
(in-scope "top.module" body+)    ; evaluate body with scope prefix
(in-scopes scope-list body+)     ; iterate over multiple scopes
(all-scopes)                     ; returns list of all scopes in waveform
```

---

## 9. Aliases

```wal
(alias myClk top.module.clk)   ; myClk now refers to top.module.clk
myClk                           ; reads top.module.clk at current INDEX
(unalias myClk)                 ; remove alias
```

Aliases are **group-aware**: inside a group, the group prefix is appended to the alias.

---

## 10. Lists

```wal
(list 1 2 3)           ; → (1 2 3)
(first xs)             ; first element
(second xs)            ; second element
(last xs)              ; last element
(rest xs)              ; all but first
(in x xs)              ; → #t if x is in xs
(length xs)            ; number of elements
(sum xs)               ; sum of all elements
(average xs)           ; mean
(min xs)               ; smallest element
(max xs)               ; largest element
(map f xs)             ; apply f to each element, return list
(fold f init xs)       ; reduce: fold f over xs with initial value init
(zip xs ys)            ; zip two lists together
```

---

## 11. Arrays (Hash Maps)

```wal
(array)                          ; create empty array (key-value map)
(seta arr key value)             ; set key → value
(geta arr key)                   ; get value at key
(geta/default arr key default)   ; get value, return default if missing
(dela arr key)                   ; delete key
(mapa f arr)                     ; apply f to each (key, value) pair
```

---

## 12. Arithmetic

```wal
(+ a b c ...)     ; addition (variadic)
(- a b c ...)     ; subtraction
(* a b c ...)     ; multiplication (variadic)
(/ a b)           ; division (float result)
(** base exp)     ; exponentiation
```

---

## 13. Types and Predicates

```wal
(atom? x)         ; true if x is an atom (not a list)
(int? x)          ; true if integer
(string? x)       ; true if string
(symbol? x)       ; true if symbol
(list? x)         ; true if list
(boolean? x)      ; true if boolean
(function? x)     ; true if function

(convert/bin x)   ; convert x to binary representation
```

---

## 14. Printing and Utility

```wal
(print args*)              ; print args separated by space, with newline
(printf "fmt %d" val)     ; C-style printf format strings

(eval-file "other.wal")   ; load and execute another WAL file, merging state
(exit 0)                  ; exit with return code
```

---

## 15. CLI Usage

Install WAL via pip, then use the `wal` command:

```bash
pip install wal-lang --user
```

### Interactive REPL

Launch the REPL by running `wal` with no arguments. The prompt shows the active trace id and current INDEX:

```
$ wal
>-> (load "trace.vcd")
t0(0) >-> SIGNALS
("tb.clk" "tb.reset" "tb.data" "tb.valid" "tb.ready")
t0(0) >-> tb.clk
0
t0(0) >-> (find (= tb.valid 1))
(10 22 34 56)
t0(0) >-> (step 10)
t0(10) >-> tb.data
42
t0(10) >-> (count (&& tb.valid tb.ready))
4
t0(10) >-> (exit 0)
$
```

The REPL maintains full state between inputs — `INDEX`, defined variables, loaded traces, and aliases all persist across lines.

### Running a WAL script file

Pass a `.wal` file as the first argument to run it non-interactively:

```bash
wal analyze.wal
```

A typical script looks like:

```wal
; analyze.wal — find all protocol violations
(load "sim.vcd")

(defun violation? []
  (&& tb.valid (! tb.ready) (> tb.data 0xff)))

(let ([hits (find (violation?))])
  (if (= (length hits) 0)
    (print "No violations found")
    (do
      (print "Violations at indices:" hits)
      (whenever (violation?)
        (printf "  [%d] data=0x%x\n" INDEX tb.data)))))

(exit 0)
```

Run it:

```bash
$ wal analyze.wal
Violations at indices: (47 203 891)
  [47] data=0x1a3
  [203] data=0x200
  [891] data=0x10f
```

### Passing a waveform directly on the command line

You can load a waveform at startup without writing `(load ...)` in the script:

```bash
wal -f sim.vcd analyze.wal
```

### FST waveform support

Install the optional `pylibfst` package to enable FST support, then use it the same way:

```bash
pip install pylibfst --user
wal -f sim.fst analyze.wal
```

---

## 16. Common Debugging Patterns

### Find all clock edges where a condition holds
```wal
(find (&& clk (! reset) valid ready))
```

### Count handshake transactions
```wal
(count (&& valid ready))
```

### Print data values during active transactions
```wal
(whenever (&& clk valid ready)
  (print INDEX "data=" data "addr=" addr))
```

### Find where a signal first goes high
```wal
(first (find (= error_flag 1)))
```

### Compute average latency between req and ack
```wal
(let ([req-times (find req)]
      [ack-times (find ack)])
  (average (map (fn [pair] (- (second pair) (first pair)))
                (zip req-times ack-times))))
```

### Generic analysis across all handshake interfaces
```wal
(let ([ifaces (groups "valid" "ready")])
  (in-groups ifaces
    (print CG "transactions:" (count (&& clk #valid #ready)))))
```

### Lookahead without losing position
```wal
(timeframe
  (while (! ack) (step))
  (print "ack arrived at" INDEX))
; back to original INDEX after this
```

### Check signal value at previous cycle
```wal
(whenever (&& clk (= state 2) (= state@-1 1))
  (print "Transition 1→2 at" INDEX))
```

---

## 17. Special Variables

| Variable | Description |
|----------|-------------|
| `INDEX`  | Current time index in the waveform |
| `SIGNALS` | List of all signal names in loaded waveform |
| `CG`     | Current Group (set inside `in-groups`) |

---

## Key Concepts Summary

- **Everything is an S-expression** in prefix notation: `(fn arg1 arg2)`
- **Signals are variables** — reading `tb.clk` gives its value at `INDEX`
- **`find`** returns a list of all matching time indices (like `grep` for waveforms)
- **`whenever`** is a forEach over all matching time indices
- **`@` operator** lets you look forward/backward: `signal@-1` reads one step earlier
- **`timeframe`** is like a save/restore of INDEX — safe lookahead
- **Groups + `#`** enable generic code that works across structurally similar signal sets
- WAL supports VCD and FST waveform formats (FST requires `pylibfst`)
- Interactive REPL: run `wal` with no arguments; run scripts with `wal script.wal`
- Online playground: https://app.wal-lang.org
