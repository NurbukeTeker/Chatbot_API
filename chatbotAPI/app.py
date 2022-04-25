from flask import Flask, render_template, request, jsonify,redirect, url_for
import model

app = Flask(__name__)


@app.route('/message', methods = ['POST'])
def reply():
    request_msg = request.get_json()  
    question = request_msg["message"]
    domain, domain_name = model.getDomainPrediction(question)
    print(domain_name) 
    return redirect(url_for(domain_name, message=question))

     

@app.route('/atis/<message>',  methods=['GET','POST'])
def atis_api(message):
    print("ATIS messaage")
    answer = model.getATISModel(message)
    answer =  "ATIS "+ str(answer)
    print(answer)
    return jsonify( { '$intent_name': answer}) 
     

@app.route('/acid/<message>', methods=['GET','POST'])
def acid_api(message):
    answer = model.getACIDModel(message)
    answer =  "ACID "+ str(answer)
    return jsonify( { '$intent_name': answer}) 
     


@app.route('/clinc/<message>',methods=['GET','POST'])
def clinc_api(message):
    answer = model.getCLINCModel(message)
    answer =  "CLINC "+ str(answer)
    return jsonify( { '$intent_name': answer}) 
     


@app.route('/banking/<message>', methods=['GET','POST'])
def banking_api(message):
    answer = model.getBANKModel(message)
    answer =  "Banking "+ str(answer)
    return jsonify( { '$intent_name':answer}) 
     

@app.route("/")
def index():
    return   "<p>ChatbotAPI</p>"


if (__name__ == "__main__"):         
    app.run(host="localhost", port = 5000)
