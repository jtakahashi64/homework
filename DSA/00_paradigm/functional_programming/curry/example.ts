function amount(a: number) {
  return (b: number) => {
    return a + b;
  };
}

const add = amount(100);

console.log(add(100)); // 200
console.log(add(100)); // 200
