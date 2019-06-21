module shifter (
    exp_A_large,
    eA_eB_abs, // Absolute difference between exponentA and exponentB
    in_mantissa_A,
    in_mantissa_B,
    out_mantissa_A,
    out_mantissa_B
);

// Parameters
parameter MANTISSA = 11;
parameter EXPONENT = 5;

// IO definition
input                exp_A_large;
input [EXPONENT-1:0] eA_eB_abs;
input [MANTISSA-1:0] in_mantissa_A;
input [MANTISSA-1:0] in_mantissa_B;

output [MANTISSA-1:0] out_mantissa_A;
output [MANTISSA-1:0] out_mantissa_B;

// Internal Wires
wire [MANTISSA-1:0] mod_mantissa_A;
wire [MANTISSA-1:0] mod_mantissa_B;

// Assignments
assign mod_mantissa_A = in_mantissa_A[MANTISSA-1]? (({MANTISSA{1'b1}}<<(MANTISSA-eA_eB_abs)) | (in_mantissa_A >> eA_eB_abs)) : (in_mantissa_A >> eA_eB_abs);
assign mod_mantissa_B = in_mantissa_B[MANTISSA-1]? (({MANTISSA{1'b1}}<<(MANTISSA-eA_eB_abs)) | (in_mantissa_B >> eA_eB_abs)) : (in_mantissa_B >> eA_eB_abs);

assign out_mantissa_A = exp_A_large ? in_mantissa_A : mod_mantissa_A;
assign out_mantissa_B = exp_A_large ? mod_mantissa_B : in_mantissa_B;

endmodule

/*
add wave -position insertpoint  \
sim:/shifter/MANTISSA \
sim:/shifter/EXPONENT \
sim:/shifter/exp_A_large \
sim:/shifter/eA_eB_abs \
sim:/shifter/in_mantissa_A \
sim:/shifter/in_mantissa_B \
sim:/shifter/out_mantissa_A \
sim:/shifter/out_mantissa_B \
sim:/shifter/mod_mantissa_A \
sim:/shifter/mod_mantissa_B
force -freeze sim:/shifter/in_mantissa_A 11101101010 0
force -freeze sim:/shifter/in_mantissa_B 00110101101 0
force -freeze sim:/shifter/exp_A_large 1 0
force -freeze sim:/shifter/eA_eB_abs 00100 0
run 100ns
*/