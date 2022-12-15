# This Python file uses the following encoding: utf-8
import sys
from PyQt5.QtWidgets import QApplication
from MainDashboard import MainDashboard

if __name__ == "__main__":
    app = QApplication([])
    window = MainDashboard()
    window.show()
    sys.exit(app.exec_())
