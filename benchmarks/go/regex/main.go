// TinyGo regex benchmark.
//
// This benchmark tests regular expression matching performance using
// the standard library regexp package.

package main

import (
	"fmt"
	"os"
	"regexp"
)

//go:wasmimport bench start
func benchStart()

//go:wasmimport bench end
func benchEnd()

func main() {
	// Read the input text file
	// Workaround for Go WASI preopen bug: Try multiple path strategies (see benchmarks/go/WASI-ISSUE.md)
	var data []byte
	var err error

	// Strategy 1: Relative path (works with TinyGo)
	data, err = os.ReadFile("regex.input")
	if err != nil {
		// Strategy 2: Absolute path
		data, err = os.ReadFile("/regex.input")
		if err != nil {
			// Strategy 3: Try to detect actual working directory
			if wd, wdErr := os.Getwd(); wdErr == nil {
				data, err = os.ReadFile(wd + "/regex.input")
			}
		}
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading input: %v\n", err)
		os.Exit(1)
	}
	text := string(data)

	// Use the same patterns as the Rust benchmark
	emailPattern := `[\w\.+-]+@[\w\.-]+\.[\w\.-]+`
	uriPattern := `[\w]+://[^/\s?#]+[^\s?#]+(?:\?[^\s#]*)?(?:#[^\s]*)?`
	ipPattern := `(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9])`

	benchStart()

	emails := countMatches(text, emailPattern)
	uris := countMatches(text, uriPattern)
	ips := countMatches(text, ipPattern)

	benchEnd()

	fmt.Fprintf(os.Stderr, "[regex] found %d emails\n", emails)
	fmt.Fprintf(os.Stderr, "[regex] found %d URIs\n", uris)
	fmt.Fprintf(os.Stderr, "[regex] found %d IPs\n", ips)
}

func countMatches(text, pattern string) int {
	re := regexp.MustCompile(pattern)
	return len(re.FindAllString(text, -1))
}
