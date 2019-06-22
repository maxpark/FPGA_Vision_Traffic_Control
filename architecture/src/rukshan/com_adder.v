module  com_adder(
    in_A,
    in_B,
    rstn,
    out_ex,
    out_m
);

// Paramters
localparam MANTISSA    = 11; 
localparam EXPONENT   = 5;
localparam DATA_WIDTH = MANTISSA + EXPONENT;

// IO definition
input  [DATA_WIDTH-1:0] in_A;
input  [DATA_WIDTH-1:0] in_B;
input                   rstn;

output [MANTISSA-1:0]    out_m;
output [EXPONENT-1:0]   out_ex;


// Internal Wires
wire [MANTISSA-1:0] mantissa_A;
wire [MANTISSA-1:0] mantissa_B;
wire [MANTISSA-1:0] mA_plus_mB;
wire [MANTISSA-1:0] shftr_mantissa_A_out;
wire [MANTISSA-1:0] shftr_mantissa_B_out;
wire [EXPONENT-1:0] exponent_A; 
wire [EXPONENT-1:0] exponent_B; 
wire [EXPONENT-1:0] eA_eB; 
wire [EXPONENT-1:0] eB_eA; 
wire [EXPONENT-1:0] max_eB_eA;
wire [EXPONENT-1:0] en;

// Shifter wires
wire [EXPONENT-1:0] eA_eB_abs;
wire                exp_A_large;

// Assignments
assign {mantissa_A,exponent_A} = in_A;
assign {mantissa_B,exponent_B} = in_B;
assign eA_eB                   = exponent_A - exponent_B;
assign eB_eA                   = exponent_B - exponent_A;
assign exp_A_large             = eA_eB[EXPONENT-1] ? 1'b0 : 1'b1;
assign eA_eB_abs               = exp_A_large ? eA_eB : eB_eA ;
assign mA_plus_mB              = shftr_mantissa_A_out + shftr_mantissa_B_out;
assign max_eB_eA               = exp_A_large ? exponent_A : exponent_B;
assign out_ex                  = max_eB_eA - en;

// Instantiations
shifter shifter(
    .exp_A_large(exp_A_large),
    .eA_eB_abs(eA_eB_abs), // Absolute difference between exponentA and exponentB
    .in_mantissa_A(mantissa_A),
    .in_mantissa_B(mantissa_B),
    .out_mantissa_A(shftr_mantissa_A_out),
    .out_mantissa_B(shftr_mantissa_B_out)
);

norm2 norm(
    .in_mantissa(mA_plus_mB),
    .out_mantissa(out_m),
    .rstn(rstn),
    .en_out(en)
);


endmodule

/*
add wave -position insertpoint  \
sim:/com_adder/MANTISSA \
sim:/com_adder/EXPONENT \
sim:/com_adder/DATA_WIDTH \
sim:/com_adder/in_A \
sim:/com_adder/in_B \
sim:/com_adder/rstn \
sim:/com_adder/out_m \
sim:/com_adder/out_ex \
sim:/com_adder/mantissa_A \
sim:/com_adder/mantissa_B \
sim:/com_adder/mA_plus_mB \
sim:/com_adder/shftr_mantissa_A_out \
sim:/com_adder/shftr_mantissa_B_out \
sim:/com_adder/exponent_A \
sim:/com_adder/exponent_B \
sim:/com_adder/eA_eB \
sim:/com_adder/eB_eA \
sim:/com_adder/max_eB_eA \
sim:/com_adder/en \
sim:/com_adder/eA_eB_abs \
sim:/com_adder/exp_A_large

force -freeze sim:/com_adder/in_A 0110110110101011 0
force -freeze sim:/com_adder/in_B 0100110111000110 0
force -freeze sim:/com_adder/rstn 1 0
*/