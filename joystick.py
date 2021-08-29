import pygame

BLACK = pygame.Color('black')
WHITE = pygame.Color('white')

class TextPrint(object):
    def __init__(self):
        self.reset()
        self.font = pygame.font.Font(None, 20)

    def tprint(self, screen, textString):
        textBitmap = self.font.render(textString, True, BLACK)
        screen.blit(textBitmap, (self.x, self.y))
        self.y += self.line_height

    def reset(self):
        self.x = 10
        self.y = 10
        self.line_height = 15

    def indent(self):
        self.x += 10

    def unindent(self):
        self.x -= 10


pygame.init()
pygame.joystick.init()

textPrint = TextPrint()
screen = pygame.display.set_mode((500, 700))

joystick = pygame.joystick.Joystick(0)
joystick.init()

buttons = joystick.get_numbuttons()
for i in range(buttons):
    button = joystick.get_button(i)
    print("Button {:>2} value: {}".format(i, button))
                     
