const readline = require('readline');
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

const getMinTime = (a, v, l, d, w) => {
    let t1, t2, t3, t4, t5, t6, t7, t8, t9;
    let v1 = Math.sqrt(2 * a * d);
    let v2 = Math.sqrt(2 * a * (l - d));
    if (v < w) {
        if (v < v1 && v < v2) {
            t1 = v / a;
            t2 = (l - a * t1 * t1 / 2) / v;
            return t1 + t2;
        } else if (v1 < v2) {
            t1 = v1 / a;
            t2 = (l - a * t1 * t1 / 2) / v1;
            return t1 + t2;
        } else {
            t1 = v2 / a;
            t2 = (l - a * t1 * t1 / 2) / v2;
            return t1 + t2;
        }
    } else {
        if (v1 < w) {
            t1 = v1 / a;
            t2 = (l - a * t1 * t1 / 2) / v1;
            return t1 + t2;
        } else {
            t1 = w / a;
            t2 = (d - a * t1 * t1 / 2) / w;
            if (v2 < v) {
                t3 = v2 / a;
                t4 = (l - d - a * t3 * t3 / 2) / v2;
                return t1 + t2 + t3 + t4;
            } else {
                t3 = (v * v - w * w) / (2 * a);
                t4 = (l - d - t3) / v;
                return t1 + t2 + t3 / v + t4;
            }
        }
    }
}

rl.on('line', (line) => {
    let [a, v, l, d, w] = line.split(' ').map(Number);
    console.log(getMinTime(a, v, l, d, w).toFixed(5));
});
