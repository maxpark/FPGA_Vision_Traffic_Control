module norm(
    in_mantissa,
    out_mantissa,
    en_out
);

parameter MANTISSA = 11;
parameter EXPONENT = 5;

input  [MANTISSA-1:0] in_mantissa;
output [MANTISSA-1:0] out_mantissa;
output [EXPONENT-1:0] en_out;

reg [3:0] counter = 4'11;

assign out_mantissa;
assign en_out;

always@(in_mantissa)
begin
    for (counter = 4'd10; counter>=4'd0;counter=counter-1 ) begin
        if in_mantissa[MANTISSA-1] begin
            en_out <= 
        end
    end
end

endmodule