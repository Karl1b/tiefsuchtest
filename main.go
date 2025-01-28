package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

type OllamaRequest struct {
	Model   string  `json:"model"`
	Prompt  string  `json:"prompt"`
	Stream  bool    `json:"stream"`
	Options Options `json:"options,omitempty"`
	NoState bool    `json:"nostate"`
}

type Options struct {
	Temperature float32 `json:"temperature,omitempty"`
}

type OllamaResponse struct {
	Model    string `json:"model"`
	Response string `json:"response"`
	Done     bool   `json:"done"`
	Error    string `json:"error,omitempty"`
}

func main() {

	task1 := "What is the smallest integer whose square is between 15 and 30?" // -5

	task2 := `If we lay 5 shirts out in the sun and it takes 4 hours to dry, how long would 20 shirts take to dry if there is enough space?` // 4 hours

	task3 := `How many times does the letter R appear in the word "strawberry"?` // 3

	task4 := `Explain classes in python using the example of cars. Show a code example with it!`

	tasks := []string{task1, task2, task3, task4}

	for i, t := range tasks {

		fmt.Printf("\n\n ### TASK %v ### \n\n", i)

		fmt.Printf("Beginne: %v mit Modell %v\n", t, "llama3:8b")
		start := time.Now()
		useLstd(t)
		fmt.Printf("\nAbgeschlossen in: %v s\n\n", time.Since(start).Seconds())

		fmt.Printf("Beginne: %v mit Modell %v\n", t, "deepseek-r1:8b")
		start = time.Now()
		useTiefSuch(t)
		fmt.Printf("\nAbgeschlossen in: %v s\n\n", time.Since(start).Seconds())

		fmt.Printf("Beginne: %v mit Modell %v\n", t, "llama3:8b mit Plan")
		start = time.Now()
		useStdMitPlan(t)
		fmt.Printf("\nAbgeschlossen in: %v s\n\n", time.Since(start).Seconds())

	}

}

func useStdMitPlan(task string) {

	newTask := fmt.Sprintf("%s \n %s", thinkP, task)
	result, err := callOllama(newTask, "llama3:8b", 0.7)
	if err != nil {
		fmt.Printf("Error with llama3:8b: %v\n", err)
		return
	}
	fmt.Printf("llama3:8b PLAN Response: %s\n", result)

	solveTask := fmt.Sprintf("%s \n This is the Plan: %s \n This is the actual task:\n %s", solveP, result, task)

	result, err = callOllama(solveTask, "llama3:8b", 0.7)
	if err != nil {
		fmt.Printf("Error with llama3:8b: %v\n", err)
		return
	}
	fmt.Printf("llama3:8b FINAL Response: %s\n", result)

}

func useLstd(task string) {

	result, err := callOllama(task, "llama3:8b", 0.7)
	if err != nil {
		fmt.Printf("Error with llama3:8b: %v\n", err)
		return
	}
	fmt.Printf("llama3:8b Response: %s\n", result)

}

func useTiefSuch(task string) {

	result, err := callOllama(task, "deepseek-r1:8b", 0.7)
	if err != nil {
		fmt.Printf("Error with deepseek-r1:8b: %v\n", err)
		return
	}
	fmt.Printf("deepseek-r1:8b Response: %s\n", result)

}

func callOllama(prompt, model string, temp float32) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Second) // 500 sec is needed!
	defer cancel()

	resultChan := make(chan string)
	errChan := make(chan error)

	go func() {
		request := OllamaRequest{
			Model:   model,
			Prompt:  prompt,
			Stream:  false,
			NoState: true,
			Options: Options{
				Temperature: temp,
			},
		}

		jsonData, err := json.Marshal(request)
		if err != nil {
			errChan <- fmt.Errorf("failed to marshal request: %v", err)
			return
		}

		req, err := http.NewRequestWithContext(ctx, "POST", "http://localhost:11434/api/generate", bytes.NewBuffer(jsonData))
		if err != nil {
			errChan <- fmt.Errorf("failed to create request: %v", err)
			return
		}
		req.Header.Set("Content-Type", "application/json")

		client := &http.Client{}
		resp, err := client.Do(req)
		if err != nil {
			errChan <- fmt.Errorf("failed to send request: %v", err)
			return
		}
		defer resp.Body.Close()

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			errChan <- fmt.Errorf("failed to read response: %v", err)
			return
		}

		var ollamaResp OllamaResponse
		if err := json.Unmarshal(body, &ollamaResp); err != nil {
			errChan <- fmt.Errorf("failed to unmarshal response: %v", err)
			return
		}

		if ollamaResp.Error != "" {
			errChan <- fmt.Errorf("ollama error: %s", ollamaResp.Error)
			return
		}

		resultChan <- ollamaResp.Response
	}()

	select {
	case <-ctx.Done():
		return "", errors.New("operation timed out")
	case err := <-errChan:
		return "", err
	case result := <-resultChan:
		return result, nil
	}
}

var thinkP string
var solveP string

func init() {

	thinkP = `You are a extremly smart planer.
	Do not solve the task provided,
	instead list all steps that are needed in detail to get to the correct solution.`

	solveP = `You are a genious solver and critic thinker.
	Solve the taks by using the thinking steps above, but question them critically, respond only with the correct answer.`
}
