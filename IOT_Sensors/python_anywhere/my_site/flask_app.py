from flask import Flask, jsonify, request, render_template
import mysql.connector
import datetime

app = Flask(__name__)

# Database connection
db = mysql.connector.connect(
    host="bhatianamrata.mysql.pythonanywhere-services.com",
    user="bhatianamrata",
    password="X(mhy&A?)T",
    database="bhatianamrata$test1"
)

# Route to handle POST requests for posting data
@app.route('/api/post_data', methods=['POST'])
def post_data():
    try:
        # Parse incoming JSON data
        data = request.json
        date_str = data['date']
        time_str = data['time']
        temperature = data['temperature']
        humidity = data['humidity']

        # Convert date_str to YYYY-MM-DD format
        date_obj = datetime.datetime.strptime(date_str, '%m/%d/%Y')
        date_str_mysql = date_obj.strftime('%Y-%m-%d')

        # Insert data into database
        cursor = db.cursor()
        query = "INSERT INTO sensor_data (date, time, temperature, humidity) VALUES (%s, %s, %s, %s)"
        cursor.execute(query, (date_str_mysql, time_str, temperature, humidity))
        db.commit()
        cursor.close()

        return jsonify({"message": "Data posted successfully"}), 200

    except Exception as e:
        return str(e), 500  # Return error message and 500 status code

# Route to retrieve data (GET request)
@app.route('/api/data', methods=['GET'])
def get_data():
    try:
        cursor = db.cursor(dictionary=True)
        query = "SELECT * FROM sensor_data ORDER BY date DESC, time DESC LIMIT 10"
        cursor.execute(query)
        data = cursor.fetchall()
        cursor.close()

        # Convert the data to a JSON serializable format
        for row in data:
            if isinstance(row['time'], datetime.timedelta):
                row['time'] = (datetime.datetime.min + row['time']).time().strftime('%H:%M:%S')
            row['date'] = row['date'].strftime('%Y-%m-%d')

        return jsonify(data)

    except Exception as e:
        return str(e), 500  # Return error message and 500 status code

# Route to retrieve data by date
@app.route('/api/data_by_date', methods=['GET'])
def get_data_by_date():
    try:
        date_str = request.args.get('date')
        if not date_str:
            return jsonify([])

        cursor = db.cursor(dictionary=True)
        query = "SELECT * FROM sensor_data WHERE date = %s ORDER BY time"
        cursor.execute(query, (date_str,))
        data = cursor.fetchall()
        cursor.close()

        # Convert the data to a JSON serializable format
        for row in data:
            row['time'] = row['time'].strftime('%H:%M:%S')
            row['date'] = row['date'].strftime('%Y-%m-%d')

        return jsonify(data)

    except Exception as e:
        return str(e), 500

# Route for rendering HTML template (GET request)
@app.route('/')
def index():
    try:
        cursor = db.cursor()
        query = "SELECT * FROM sensor_data ORDER BY date DESC, time DESC LIMIT 10"
        cursor.execute(query)
        data = cursor.fetchall()
        cursor.close()
        return render_template('index.html', data=data)

    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
