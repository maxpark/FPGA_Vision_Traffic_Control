module  com_adder(
    in_A,
    in_B,
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
assign max_eB_eA               = exp_A_large ? exponentA : exponentB;
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

norm norm(
    .in_mantissa(mA_plus_mB),
    .out_mantissa(out_m),
    .en_out(en)
);


endmodule