

class Perceptron:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def forward_pass(self, single_input):
        result = 0
        for i in range(0, len(self.w)):
            result += self.w[i] * single_input[i]
        result += self.b

        if result > 0:
            return 1
        else:
            return 0

    def vectorized_forward_pass(self, input_matrix):
        result = input_matrix.dot(self.w)
        return result + self.b > 0

    def train_on_single_example(self, example, y):
        answer = self.forward_pass(example)
        diff = y - answer
        self.b += diff
        self.w += example * diff
        return abs(diff)

    def train_until_convergence(self, input_matrix, y, max_steps=1e8):
        i = 0
        errors = 1
        while errors and i < max_steps:
            i += 1
            errors = 0
            for example, answer in zip(input_matrix, y):
                example = example.reshape((example.size, 1))
                error = self.train_on_single_example(example, answer)
                errors += int(error)