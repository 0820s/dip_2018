import web
from test_cuda import upscale#
urls = ('/dip', 'Upload')
render = web.template.render('templates/',)
    
class Upload:
    def GET(self):
        web.header("Content-Type","text/html; charset=utf-8")
        return render.index(filename='',h='',w='')

    def POST(self):
        x = web.input(myfile={})
        if 'myfile' in x: 
            filepath=x.myfile.filename.replace('\\','/') 
            filename=filepath.split('/')[-1] 
            fout = open('./static/'+ filename,'wb') 
            fout.write(x.myfile.file.read()) 
            fout.close() 
            
            (h,w)=upscale(filename)# replace this sentence 

        return render.index(filename=filename,h=h,w=w)

if __name__ == "__main__":
   app = web.application(urls, globals()) 
   app.run()