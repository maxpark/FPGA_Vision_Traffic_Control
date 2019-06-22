module norm2(
    in_mantissa,
    out_mantissa,
    rstn,
    en_out
);

parameter MANTISSA = 11;
parameter EXPONENT = 5;

input  [MANTISSA-1:0] in_mantissa;
input                 rstn;

output [MANTISSA-1:0] out_mantissa;
output [EXPONENT-1:0] en_out;

reg [MANTISSA-1:0] out_mantissa_reg = {MANTISSA{1'b0}};
reg [EXPONENT-1:0] en_out_reg       = {EXPONENT{1'b0}};


assign out_mantissa = out_mantissa_reg;
assign en_out       = en_out_reg;

always@(*)
begin
    if (~rstn) 
    begin
        out_mantissa_reg <= {MANTISSA{1'b0}};
        en_out_reg       <= {EXPONENT{1'b0}};
    end else if (in_mantissa[MANTISSA-1]) // Negative
    begin
        casez (in_mantissa)
        11'b10?????????:
            begin
                en_out_reg       <= 5'd0;
                out_mantissa_reg <= (in_mantissa);
            end
        11'b110????????:
            begin
                en_out_reg       <= 5'd1;
                out_mantissa_reg <= {(in_mantissa[9:0]),{1{1'b1}}};
            end
        11'b1110???????:
            begin
                en_out_reg       <= 5'd2;
                out_mantissa_reg <= {(in_mantissa[8:0]),{2{1'b1}}};
            end
        11'b11110??????:
            begin
                en_out_reg       <= 5'd3;
                out_mantissa_reg <= {(in_mantissa[7:0]),{3{1'b1}}};
            end
        11'b111110?????:
            begin
                en_out_reg       <= 5'd4;
                out_mantissa_reg <= {(in_mantissa[6:0]),{4{1'b1}}};
            end
        11'b1111110????:
            begin
                en_out_reg       <= 5'd5;
                out_mantissa_reg <= {(in_mantissa[5:0]),{5{1'b1}}};
            end
        11'b11111110???:
            begin
                en_out_reg       <= 5'd6;
                out_mantissa_reg <= {(in_mantissa[4:0]),{6{1'b1}}};
            end
        11'b111111110??:
            begin
                en_out_reg       <= 5'd7;
                out_mantissa_reg <= {(in_mantissa[3:0]),{7{1'b1}}};
            end
        11'b1111111110?:
            begin
                en_out_reg       <= 5'd8;
                out_mantissa_reg <= {(in_mantissa[2:0]),{8{1'b1}}};
            end
        11'b11111111110:
            begin
                en_out_reg       <= 5'd9;
                out_mantissa_reg <= {(in_mantissa[1:0]),{9{1'b1}}};
            end
        11'b11111111111:
            begin
                en_out_reg       <= 5'd0;
                out_mantissa_reg <= (in_mantissa);
            end
        default:
            begin
                en_out_reg       <= 5'd0;
                out_mantissa_reg <= (in_mantissa);
            end
        endcase
    end else begin // Positive
        casez(in_mantissa)
        11'b01?????????:
            begin
                en_out_reg       <= 5'd0;
                out_mantissa_reg <= (in_mantissa);
            end
        11'b001????????:
            begin
                en_out_reg       <= 5'd1;
                out_mantissa_reg <= (in_mantissa << 1);
            end
        11'b0001???????:
            begin
                en_out_reg       <= 5'd2;
                out_mantissa_reg <= (in_mantissa << 2);
            end
        11'b00001??????:
            begin
                en_out_reg       <= 5'd3;
                out_mantissa_reg <= (in_mantissa << 3);
            end
        11'b000001?????:
            begin
                en_out_reg       <= 5'd4;
                out_mantissa_reg <= (in_mantissa << 4);
            end
        11'b0000001????:
            begin
                en_out_reg       <= 5'd5;
                out_mantissa_reg <= (in_mantissa << 5);
            end
        11'b00000001???:
            begin
                en_out_reg       <= 5'd6;
                out_mantissa_reg <= (in_mantissa << 6);
            end
        11'b000000001??:
            begin
                en_out_reg       <= 5'd7;
                out_mantissa_reg <= (in_mantissa << 7);
            end
        11'b0000000001?:
            begin
                en_out_reg       <= 5'd8;
                out_mantissa_reg <= (in_mantissa << 8);
            end
        11'b00000000001:
            begin
                en_out_reg       <= 5'd9;
                out_mantissa_reg <= (in_mantissa << 9);
            end
        11'b00000000000:
            begin
                en_out_reg       <= 5'd0;
                out_mantissa_reg <= (in_mantissa);
            end
        default:
            begin
                en_out_reg       <= 5'd0;
                out_mantissa_reg <= (in_mantissa);
            end
        endcase
    end
end

endmodule