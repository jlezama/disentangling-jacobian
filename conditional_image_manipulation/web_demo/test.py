javascript = '<script>'

for i in range(40):
    javascript += """
var slider{i} = document.getElementById("myRange_{i}");
var output{i} = document.getElementById("value_{i}");
output{i}.innerHTML = slider{i}.value;
slider{i}.oninput = function() {{  output{i}.innerHTML = this.value;}} 
slider{i}.onmouseup = function(){{document.getElementById("sliders_form").submit();}}; 
""".format(i=i)


print javascript
        
