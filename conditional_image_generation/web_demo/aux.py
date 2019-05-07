attr_dict = {
     0: '5 o Clock shadow', 
     1: 'Arched Eyebrows', 
     2: 'Attractive', 
     3: 'Bags Under Eyes', 
     4: 'Bald', 
     5: 'Bangs', 
     6: 'Big Lips', 
     7: 'Big Nose', 
     8: 'Black Hair', 
     9: 'Blond Hair', 
    10: 'Blurry', 
    11: 'Brown hair', 
    12: 'Bushy Eyebrows', 
    13: 'Chubby', 
    14: 'Double Chin', 
    15: 'Eyeglasses', 
    16: 'Goatee', 
    17: 'Gray Hair', 
    18: 'Heavy Makeup', 
    19: 'High Cheekbones', 
    20: 'Male', 
    21: 'Mouth Slightly Open', 
    22: 'Mustache', 
    23: 'Narrow Eyes', 
    24: 'No Beard', 
    25: 'Oval Face', 
    26: 'Pale Skin', 
    27: 'Pointy nose', 
    28: 'Receding Hairline', 
    29: 'Rosy Cheeks', 
    30: 'Sideburns', 
    31: 'Smiling', 
    32: 'Straight Hair', 
    33: 'Wavy Hair', 
    34: 'Wearing Earrings', 
    35: 'Wearing Hat', 
    36: 'Wearing Lipstick', 
    37: 'Wearing Necklace', 
    38: 'Wearing Necktie', 
    39: 'Young', 
    }


################################################################################
def create_html_result(outfname, y_pred, params):
    # create html table
    rows1 = ''
    rows2 = ''
    for i in range(40):
        row = '<tr><td style="font-size:14"> %s </td><td >  <div class="slidecontainer">  <input type="range" min="-40" max="40" step="0.1" class="slider" id="myRange_%i" name="attr_%i" value="%2.2f"> </div>  </td> <td id="value_%i" style="width:40"> %2.2f</td></tr>' % (attr_dict[i], i,i, y_pred[i], i, y_pred[i])
        if i<20:
            rows1 += row
        else:
            rows2 += row

    model_description = params.model_path

        
    table_txt = """
    <div  style="margin-left:2cm;margin-top:3cm;font-family:'Trebuchet MS'">
    <p style="text-align:center;font-size:28px">%s</p>
    <table> 
      <tr>
        <td>
          <table border=0  >%s</table>
        </td>
         
        <td>
          <table border=0 >%s</table>
        </td>
        <td style="text-align:center;font-size:28px">
 Original / Reconstruction / Manipulation <br><br>
          <img src="%s" id="result_img">
        </td>
      </tr>
    </table>
</div>
    """ % (model_description, rows1, rows2, outfname)




    javascript = '<script>'

    for i in range(40):
        javascript += """
        var slider{i} = document.getElementById("myRange_{i}");
        var output{i} = document.getElementById("value_{i}");
        output{i}.innerHTML = slider{i}.value;
        slider{i}.oninput = function() {{  output{i}.innerHTML = this.value;}} 
        slider{i}.onmouseup = function(){{document.getElementById("sliders_form").submit();}}; 
        """.format(i=i)

    javascript += '</script>'
    
    html = """
        <html><body> <form action='/get_image' method=GET id="img_id_form"> Image ID: <input type="text" name="fname" value="%i">
         <input type="submit" value="Submit">
</form>
         <br> <hr> <br>
    <form action='.' method=POST id="sliders_form">
         %s
         </form>
         %s
        </body></html>
    """ % (params.offset, table_txt, javascript)

    return html

